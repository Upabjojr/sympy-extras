"""The SymPy face of the Risch package: SymPy objects in and out, the
answers checked by differentiation."""
from __future__ import annotations

from sympy import Integral, Rational, cancel, diff, exp, log, symbols, tan, together
from sympy.core.expr import Expr

from sympy_extras._typing_sympy import as_expr
from sympy_extras.integrals.risch.risch_sympy import (NonElementaryIntegral, is_nonelementary, rational_risch_de,
                                                      risch_antiderivative, risch_integrate)

x = symbols('x')


def _antiderivative(F: Expr, f: Expr) -> bool:
    return cancel(together(diff(F, x) - f)) == 0


def test_face_risch_integrate() -> None:
    f = as_expr(exp(x)*log(exp(x) + 1))
    F = risch_integrate(f, x)
    assert isinstance(F, Expr) and not F.has(Integral) and _antiderivative(F, f)
    g = as_expr(exp(-x**2))
    G = risch_integrate(g, x)
    assert isinstance(G, NonElementaryIntegral) and G.function == g
    ans, i = risch_integrate(as_expr((2*log(x)**2 - log(x) - x**2)/(log(x)**3 - x**2*log(x))), x,
                             separate_integral=True)
    assert isinstance(i, NonElementaryIntegral) and i.function == 1/log(x)
    # floats: SymPy's domain RR, the coefficients of the answer as floats
    h = as_expr((-60*exp(x) - Rational(96, 5)*exp(4*x))*exp(4*x))
    assert risch_integrate(h, x) == -Rational(12, 5)*exp(8*x) - 12*exp(5*x)


def test_face_antiderivative() -> None:
    assert risch_antiderivative(tan(x)**2, x) == tan(x) - x
    assert risch_antiderivative(exp(x**2), x) is None
    assert is_nonelementary(exp(x)/x, x) is True
    assert is_nonelementary(x*exp(x), x) is False


def test_rational_risch_de() -> None:
    # y' + y/x == 2: y == x
    assert rational_risch_de(1/x, 2, x) == x
    # y' + y == 1/x has no rational solution (Ei)
    assert rational_risch_de(1, 1/x, x) is None
