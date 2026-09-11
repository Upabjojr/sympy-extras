"""Integrals of D-finite functions with known values (DLMF, Gradshteyn
and Ryzhik), by Chyzak's algorithm, each checked numerically too."""
from __future__ import annotations

from sympy import symbols, exp, besselj, sin, cos, oo, sqrt, pi, atan, legendre, sinh, cosh, simplify, S

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals.dfinite import dfinite_integral
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.telescoping import holonomic_integral

x = symbols('x')
t = symbols('t', positive=True)


def _check(F: ExprLike, a: ExprLike, b: ExprLike, expected: ExprLike) -> None:
    found = dfinite_integral(as_expr(F), x, as_expr(a), as_expr(b), t)
    assert found is not None, F
    assert simplify((found.value - as_expr(expected)).rewrite(exp)) == 0, (found.value, expected)
    assert verify_numerically(found.value, as_expr(F), x, as_expr(a), as_expr(b)) is not False


def test_bessel_and_exponential() -> None:
    # DLMF 10.22.49 with nu = 0: Integral(exp(-t x) J_0(x), (x, 0, oo)) = 1/sqrt(1 + t^2)
    _check(exp(-t * x) * besselj(0, x), 0, oo, 1 / sqrt(1 + t**2))
    # GR 6.631.4: Integral(x exp(-x^2) J_0(t x), (x, 0, oo)) = exp(-t^2/4)/2
    _check(x * exp(-x**2) * besselj(0, t * x), 0, oo, exp(-t**2 / 4) / 2)


def test_trigonometric() -> None:
    # GR 3.941.1: Integral(exp(-t x) sin(x)/x, (x, 0, oo)) = acot(t)
    _check(exp(-t * x) * sin(x) / x, 0, oo, pi / 2 - atan(t))
    # GR 3.952.8-type: Integral(x^2 exp(-x^2) cos(2 t x), (x, 0, oo)) = sqrt(pi) (1 - 2 t^2) exp(-t^2)/4
    F = x**2 * exp(-x**2) * cos(2 * t * x)
    _check(F, 0, oo, sqrt(pi) * (1 - 2 * t**2) * exp(-t**2) / 4)
    # the same by the hyperexponential algorithm: the two methods agree
    other = holonomic_integral(F, x, S.Zero, oo, t)
    ours = dfinite_integral(F, x, S.Zero, oo, t)
    assert other is not None and ours is not None and simplify(other.value - ours.value) == 0


def test_orthogonal_polynomial() -> None:
    # Integral(P_2(x) exp(t x), (x, -1, 1)) = 2 (t^2 + 3) sinh(t)/t^3 - 6 cosh(t)/t^2 (by parts twice)
    _check(legendre(2, x) * exp(t * x), -1, 1, 2 * (t**2 + 3) * sinh(t) / t**3 - 6 * cosh(t) / t**2)


def test_gaussian() -> None:
    _check(exp(-t * x**2), 0, oo, sqrt(pi) / (2 * sqrt(t)))
    _check(exp(-x**2) * exp(2 * t * x), -oo, oo, sqrt(pi) * exp(t**2))
