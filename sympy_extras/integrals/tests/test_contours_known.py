"""Known values for the three contours (Gradshteyn–Ryzhik, 7th edition,
and Whittaker–Watson), each also checked numerically."""
from __future__ import annotations

from sympy import symbols, exp, sinh, cosh, sin, cos, oo, pi, S, sqrt, simplify, Rational

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.integrals.contours import contour_integral
from sympy_extras.integrals.definite import verify_numerically

x = symbols('x')
a, b, n = symbols('a b n', positive=True)


def _check(value: ExprLike, expected: ExprLike, f: ExprLike, lo: ExprLike, hi: ExprLike,
           assumptions: Assumptions = None) -> None:
    assert simplify(as_expr(value) - as_expr(expected)) == 0, (value, expected)
    assert verify_numerically(as_expr(expected), as_expr(f), x, as_expr(lo), as_expr(hi), assumptions) is not False


def test_gradshteyn_ryzhik() -> None:
    # GR 3.311.3: Integral(exp(a x)/(1 + exp(x)), (x, -oo, oo)) = pi/sin(pi a), 0 < a < 1
    found = contour_integral(exp(a * x) / (1 + exp(x)), x, -oo, oo, a < 1)
    assert found is not None
    _check(found.value, pi / sin(pi * a), exp(a * x) / (1 + exp(x)), -oo, oo, a < 1)
    # GR 3.521.1: Integral(x/sinh(x), (x, 0, oo)) = pi**2/4, so pi**2/2 over the line
    found = contour_integral(x / sinh(x), x, -oo, oo)
    assert found is not None
    _check(found.value, pi**2 / 2, x / sinh(x), -oo, oo)
    # GR 3.511.4: Integral(1/cosh(x), (x, 0, oo)) = pi/2
    found = contour_integral(1 / cosh(x), x, -oo, oo)
    assert found is not None
    _check(found.value, pi, 1 / cosh(x), -oo, oo)
    # GR 3.241.2: Integral(x**(mu - 1)/(1 + x**nu), (x, 0, oo)) = pi/(nu sin(mu pi/nu))
    found = contour_integral(x**a / (1 + x**n), x, S.Zero, oo, a < n - 1)
    assert found is not None
    assert simplify(found.value - pi / (n * sin(pi * (a + 1) / n))) == 0
    assert verify_numerically(found.value, x**a / (1 + x**n), x, S.Zero, oo, a < n - 1) is not False
    # GR 3.737.1: Integral(sin(a x)/(x (x**2 + b**2)), (x, 0, oo)) = pi (1 - exp(-a b))/(2 b**2)
    found = contour_integral(sin(a * x) / (x * (x**2 + b**2)), x, S.Zero, oo)
    assert found is not None
    _check(found.value, pi * (1 - exp(-a * b)) / (2 * b**2), sin(a * x) / (x * (x**2 + b**2)), 0, oo)


def test_whittaker_watson_and_others() -> None:
    # Whittaker-Watson 6.24: Integral(exp(a x)/cosh(x), (x, -oo, oo)) = pi/cos(pi a/2), |a| < 1
    found = contour_integral(exp(a * x) / cosh(x), x, -oo, oo, a < 1)
    assert found is not None
    assert verify_numerically(found.value, exp(a * x) / cosh(x), x, -oo, oo, a < 1) is not False
    assert abs(float((found.value - pi / cos(pi * a / 2)).subs(a, Rational(1, 3)).evalf(20))) < 1e-15
    # Integral(x**2/cosh(x), (x, -oo, oo)) = pi**3/4 (GR 3.523.4 with the line doubled)
    found = contour_integral(x**2 / cosh(x), x, -oo, oo)
    assert found is not None
    _check(found.value, pi**3 / 4, x**2 / cosh(x), -oo, oo)
    # Integral(1/(2 cosh(x) + 1), (x, -oo, oo)) = 2 pi/(3 sqrt(3)) (GR 3.513.? / by the residues at 2 pi i/3, 4 pi i/3)
    found = contour_integral(1 / (2 * cosh(x) + 1), x, -oo, oo)
    assert found is not None
    _check(found.value, 2 * pi / (3 * sqrt(3)), 1 / (2 * cosh(x) + 1), -oo, oo)
