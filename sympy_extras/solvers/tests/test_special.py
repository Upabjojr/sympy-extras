from __future__ import annotations

from sympy import Function, Symbol, symbols, N, besselj, bessely, hyper, sqrt, Rational, Basic, S, exp
from sympy.core.expr import Expr

from sympy_extras.solvers.special import (special_solutions, bessel_solutions, whittaker_solutions,
    hypergeometric_solutions, whittaker_m)
from sympy_extras.solvers.linear_ode import dsolve_linear

x = Symbol('x')
y = Function('y')(x)
a, b, c, n = symbols('a b c n')
_VALUES: dict[Basic, Basic] = {a: Rational(3, 10), b: Rational(17, 10), c: Rational(23, 10), n: Rational(2, 5)}


def _residual(equation: Expr, solution: Expr) -> float:
    e = equation.subs(y, solution).doit().subs(_VALUES)
    return max(float(abs(N(e.subs(x, v), 20))) for v in (Rational(7, 10), Rational(13, 10), Rational(21, 10)))


def _solved(equation: Expr, count: int = 2) -> list[Expr]:
    found = special_solutions(equation, y)
    assert found is not None, equation
    assert len(found) == count, found
    for s in found:
        assert _residual(equation, s) < 1e-10, (equation, s)
    return found


def test_bessel() -> None:
    found = _solved(y.diff(x, 2) + x*y)                           # Airy
    assert found[0] == sqrt(x)*besselj(Rational(1, 3), 2*x**Rational(3, 2)/3)
    _solved(y.diff(x, 2) + (a*x + b)*y)
    found = _solved(x**2*y.diff(x, 2) + x*y.diff(x) + (x**2 - n**2)*y)
    assert found == [besselj(n, x), bessely(n, x)]
    _solved(x*y.diff(x, 2) + y.diff(x) + a*y)
    _solved(x*y.diff(x, 2) - y.diff(x) + a*y)
    _solved(4*y.diff(x, 2) + 9*x*y)
    _solved(y.diff(x, 2) - x**4*y)
    assert bessel_solutions(x**2 + a, x) is None


def test_whittaker() -> None:
    _solved(y.diff(x, 2) - (x**2 + a)*y)                          # parabolic cylinder
    _solved(y.diff(x, 2) - 2*x*y.diff(x) + a*y)                    # Hermite
    _solved(y.diff(x, 2) + a*y.diff(x) + (-b**2*x**2 - c)*y)
    _solved(x*y.diff(x, 2) + (b - x)*y.diff(x) - a*y)              # Kummer
    _solved(y.diff(x, 2) + x*y.diff(x) + (n + 1)*y)
    _solved(x*y.diff(x, 2) + (a + x)*y, count=1)                   # 2 mu integer: one solution
    assert whittaker_m(S.Zero, Rational(1, 2), x) == x*exp(-x/2)*hyper((1,), (2,), x)
    assert whittaker_solutions(-x, x) is None


def test_hypergeometric() -> None:
    found = _solved((1 - x**2)*y.diff(x, 2) - 2*x*y.diff(x) + n*(n + 1)*y, count=1)   # Legendre
    assert found[0].has(hyper((-n, n + 1), (1,), Rational(1, 2) - x/2))
    _solved(x*(1 - x)*y.diff(x, 2) + (c - (a + b + 1)*x)*y.diff(x) - a*b*y)             # Gauss
    _solved((1 - x**2)*y.diff(x, 2) - x*y.diff(x) + n**2*y)                             # Chebyshev
    _solved(x*(x - 1)*y.diff(x, 2) + (2*x - 1)*y.diff(x) + a*y, count=1)
    # three finite regular singular points with infinity regular
    _solved(4*x**2*(x - 1)**2*(x - 2)**2*y.diff(x, 2) - y)
    # four singular points (Heun) are not hypergeometric
    assert hypergeometric_solutions(1/(x**2*(x - 1)**2*(x - 2)**2) + 1/x, x) is None
    assert special_solutions((x**2 - 1)*(x - 2)*y.diff(x, 2) + y, y) is None


def test_dsolve_linear_uses_special_functions() -> None:
    found = dsolve_linear(x*y.diff(x, 2) + y.diff(x) + a*y, y)
    assert found == [besselj(0, 2*sqrt(a)*sqrt(x)), bessely(0, 2*sqrt(a)*sqrt(x))]
    found = dsolve_linear(y.diff(x, 2) + x*y, y)
    assert len(found) == 2 and all(_residual(y.diff(x, 2) + x*y, s) < 1e-10 for s in found)
