"""Tests of the rectangle, the sector and the indented real line on the
objects of the core (:mod:`sympy_extras.integrals.contours`): the values
against Gradshteyn-Ryzhik and mpmath's quadrature (independent sources),
the elimination of the triangular system, and the agreement with the
SymPy face."""
from __future__ import annotations

from typing import Callable, Optional

import mpmath

from sympy_extras.core import Integer, S, Symbol, oo, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, cosh, exp, sin, sinh
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.contours import (_solve, contour_integral, indented_integral, rectangular_integral,
                                             sector_integral)
from sympy_extras.simplify.simplify import simplify

x = Symbol('x')
a, b, n = Symbol('a', positive=True), Symbol('b', positive=True), Symbol('n', positive=True)


def _value(found: Optional[ConditionalValue]) -> Expr:
    assert found is not None
    return found.value


def _number(e: Expr) -> complex:
    return complex(e.evalf(30))


def _same(first: Expr, second: Expr) -> bool:
    if simplify(first - second) == 0:
        return True
    return abs(_number(first) - _number(second)) < 1e-20


def _real_line(f: Callable[[mpmath.mpf], mpmath.mpf]) -> complex:
    with mpmath.workdps(30):
        return complex(mpmath.quad(f, [-mpmath.inf, -1, 0, 1, mpmath.inf]))


def test_solve() -> None:
    # a lower triangular system and one which needs a row exchange
    assert _solve([[Integer(2), S.Zero], [Integer(1), Integer(3)]], [Integer(4), Integer(5)]) == [2, 1]
    assert _solve([[S.Zero, Integer(1)], [Integer(1), Integer(1)]], [Integer(2), Integer(3)]) == [1, 2]
    assert _solve([[S.Zero, S.Zero], [Integer(1), Integer(1)]], [Integer(2), Integer(3)]) is None


def test_rectangle() -> None:
    assert _same(_value(rectangular_integral(1 / cosh(x), x)), pi)
    assert _same(_value(rectangular_integral(x / sinh(x), x)), pi**2 / 2)
    assert _same(_value(rectangular_integral(x**2 / cosh(x), x)), pi**3 / 4)
    # GR 3.311.3
    found = rectangular_integral(exp(a * x) / (1 + exp(x)), x)
    assert found is not None
    assert _same(found.value, pi / sin(pi * a))
    assert found.condition == ((a > 0) & (a < 1))
    # a double pole of R: the core's residue
    assert _same(_value(rectangular_integral(1 / cosh(x)**2, x)), Integer(2))
    value = _value(rectangular_integral(1 / (2 * cosh(x) + 1), x))
    assert abs(_number(value) - _real_line(lambda t: 1 / (2 * mpmath.cosh(t) + 1))) < 1e-20
    # no decay
    assert rectangular_integral(exp(x) / (1 + exp(x)), x) is None
    # the bug: exp(a) and exp(-a) were two unrelated placeholders, so that
    # the discriminant of u**2 + 2*cosh(a)*u + 1 was no square and the
    # value a sum of logarithms of radicals; exp(k*a) is exp(a)**k, as
    # SymPy's generators are, and the value 2*a/sinh(a) (GR 3.514.1)
    assert _value(rectangular_integral(1 / (cosh(x) + cosh(a)), x)) == 2 * a / sinh(a)


def test_sector() -> None:
    found = sector_integral(x**a / (1 + x**n), x, a < n - 1)
    assert found is not None
    assert _same(found.value, pi / (n * sin(pi * (a + 1) / n)))
    assert _same(_value(sector_integral(x**2 / (3 + 2 * x**6), x)), sqrt(6) * pi / 36)
    value = _value(sector_integral(1 / (1 + x**5), x))
    with mpmath.workdps(30):
        expected = complex(mpmath.quad(lambda t: 1 / (1 + t**5), [0, 1, mpmath.inf]))
    assert abs(_number(value) - expected) < 1e-20
    assert sector_integral(1 / (1 - x**2), x) is None


def test_indented() -> None:
    assert _same(_value(indented_integral(sin(x) / x, x, -oo, oo)), pi)
    assert _same(_value(indented_integral(sin(x) / (x * (x**2 + 1)), x, 0, oo)), pi * (1 - exp(-1)) / 2)
    assert _same(_value(indented_integral(sin(2 * x) / x, x, 0, oo)), pi / 2)
    # GR 3.737.1 with a symbolic frequency and width
    assert _same(_value(indented_integral(sin(a * x) / (x * (x**2 + b**2)), x, 0, oo)),
                 pi * (1 - exp(-a * b)) / (2 * b**2))
    # the cosine does not vanish at the real poles: the integrand is singular
    assert indented_integral(cos(x) / (x**2 - 1), x, -oo, oo) is None
    assert indented_integral(sin(x) / x, x, 0, 1) is None


def test_dispatch() -> None:
    assert _same(_value(contour_integral(x / sinh(x), x, -oo, oo)), pi**2 / 2)
    assert _same(_value(contour_integral(x**a / (1 + x**n), x, 0, oo, a < n - 1)), pi / (n * sin(pi * (a + 1) / n)))
    assert _same(_value(contour_integral(sin(x) / x, x, -oo, oo)), pi)
    assert contour_integral(1 / (x**2 + 1), x, 0, 1) is None


def test_the_face_agrees() -> None:
    import sympy
    from sympy_extras.integrals import contours_sympy
    from sympy_extras._numeric_sympy import core_expr
    X = sympy.Symbol('x')
    for f, lo, hi in [(X / sympy.sinh(X), -sympy.oo, sympy.oo), (1 / (1 + X**3), 0, sympy.oo),
                      (sympy.sin(X) / (X * (X**2 + 4)), -sympy.oo, sympy.oo)]:
        face = contours_sympy.contour_integral(f, X, lo, hi)
        native = contour_integral(core_expr(f), x, core_expr(lo), core_expr(hi))
        assert face is not None and native is not None
        assert core_expr(face.value) == native.value
