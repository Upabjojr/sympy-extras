"""Parametric integrals with known values (Gradshteyn–Ryzhik, 7th
edition, [GR]; DLMF) computed by the holonomic method, each also checked
numerically."""
from __future__ import annotations

from sympy import symbols, exp, cos, sqrt, pi, oo, simplify

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.telescoping import holonomic_integral, almkvist_zeilberger

x = symbols('x')
t = symbols('t', positive=True)


def _check(F: ExprLike, a: ExprLike, b: ExprLike, expected: ExprLike) -> None:
    found = holonomic_integral(F, x, a, b, t)
    assert found is not None, F
    assert simplify(found.value - as_expr(expected)) == 0, (found, expected)
    assert verify_numerically(as_expr(expected), as_expr(F), x, as_expr(a), as_expr(b)) is not False


def test_gaussian_integrals() -> None:
    # GR 3.896.4: Integral(exp(-x**2) cos(2 t x), (x, 0, oo)) = sqrt(pi)/2 exp(-t**2)
    _check(exp(-x**2) * cos(2 * t * x), 0, oo, sqrt(pi) * exp(-t**2) / 2)
    # GR 3.323.2: Integral(exp(-x**2 + 2 t x), (x, -oo, oo)) = sqrt(pi) exp(t**2)
    _check(exp(-x**2 + 2 * t * x), -oo, oo, sqrt(pi) * exp(t**2))
    # GR 3.325: Integral(exp(-x**2 - t**2/x**2), (x, 0, oo)) = sqrt(pi)/2 exp(-2 t)
    _check(exp(-x**2 - t**2 / x**2), 0, oo, sqrt(pi) * exp(-2 * t) / 2)
    # GR 3.952.7 (a moment): Integral(x**2 exp(-x**2) cos(2 t x), (x, 0, oo)) = sqrt(pi)/4 (1 - 2 t**2) exp(-t**2)
    _check(x**2 * exp(-x**2) * cos(2 * t * x), 0, oo, sqrt(pi) * (1 - 2 * t**2) * exp(-t**2) / 4)


def test_rational_and_exponential_integrals() -> None:
    # GR 3.241.4 (n = 2): Integral(1/(x**2 + t**2)**2, (x, -oo, oo)) = pi/(2 t**3)
    _check(1 / (x**2 + t**2)**2, -oo, oo, pi / (2 * t**3))
    _check(1 / (x**2 + t**2), -oo, oo, pi / t)
    # DLMF 5.2.1 with an exponential scale: Integral(x**3 exp(-t x), (x, 0, oo)) = 6/t**4
    _check(x**3 * exp(-t * x), 0, oo, 6 / t**4)
    # GR 3.351.3: Integral(x**2 exp(-t x), (x, 0, oo)) = 2/t**3
    _check(x**2 * exp(-t * x), 0, oo, 2 / t**3)


def test_equations_match_the_closed_forms() -> None:
    # the telescoper's equation is satisfied by the known value
    from sympy import Function
    for F, value in [(exp(-x**2) * cos(2 * t * x), sqrt(pi) * exp(-t**2) / 2),
                     (1 / (x**2 + t**2)**2, pi / (2 * t**3))]:
        found = almkvist_zeilberger(F, x, t)
        assert found is not None
        I = Function('I')(t)
        assert simplify(found.operator().subs(I, value).doit()) == 0
