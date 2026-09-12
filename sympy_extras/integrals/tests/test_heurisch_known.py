"""The heuristic Risch integrator against the classical tables: the
families of Gradshteyn and Ryzhik 2.1 (rational), 2.2 (algebraic, one
square root of a quadratic), 2.3 (exponential), 2.4 (hyperbolic), 2.5
(trigonometric) and 2.7 (logarithmic), each result checked by
differentiation and, for the closed forms quoted, against the table."""
from __future__ import annotations

from typing import Optional, Sequence

from sympy import symbols, exp, sin, cos, tan, log, sqrt, asin, asinh, atan, Rational, diff, cancel, simplify, sinh, cosh
from sympy.logic.boolalg import Boolean

from sympy.core.expr import Expr

from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.integrals.conditions import numerically_equal
from sympy_extras.integrals.heurisch import heurisch_antiderivative

x, a, b = symbols('x a b')
p = symbols('p', positive=True)


def _same(F: object, G: ExprLike) -> bool:
    return isinstance(F, Expr) and simplify(F - as_expr(G)) == 0


def _checks(f: ExprLike, facts: Optional[Sequence[Boolean]] = None) -> bool:
    F = heurisch_antiderivative(as_expr(f), x)
    if F is None:
        return False
    difference = as_expr(diff(F, x) - as_expr(f))
    if cancel(difference) == 0 or simplify(difference) == 0:
        return True
    return numerically_equal(as_expr(diff(F, x)), as_expr(f), [as_boolean(c) for c in (facts or [])])


def test_gr_2_1_rational() -> None:
    # 2.103, 2.124, 2.141, 2.143, 2.148
    assert _same(heurisch_antiderivative(1 / (x**2 + p**2), x), atan(x / p) / p)
    assert _same(heurisch_antiderivative(1 / (x**2 - p**2), x), (log(x - p) - log(x + p)) / (2 * p))
    assert _same(heurisch_antiderivative(x / (x**2 + p**2)**2, x), -1 / (2 * (x**2 + p**2)))
    assert _checks(1 / (x**3 + 1)) and _checks(1 / (x**4 + 1)) and _checks(x**2 / (x**4 + 1))
    assert _checks((x**2 + 1) / (x**4 + x**2 + 1)) and _checks(1 / (x**6 + 1)) and _checks(1 / (x * (x**2 + 1)))


def test_gr_2_2_algebraic() -> None:
    # 2.261, 2.271, 2.272, 2.264: one square root of a quadratic
    assert _same(heurisch_antiderivative(1 / sqrt(x**2 + p**2), x), asinh(x / p))
    assert _same(heurisch_antiderivative(1 / sqrt(p**2 - x**2), x), asin(x / p))
    assert _same(heurisch_antiderivative(x / sqrt(x**2 + p**2), x), sqrt(x**2 + p**2))
    assert _same(heurisch_antiderivative(sqrt(p**2 - x**2), x), x * sqrt(p**2 - x**2) / 2 + p**2 * asin(x / p) / 2)
    assert _checks(x**2 / sqrt(1 - x**2), [x > -1, x < 1]) and _checks(sqrt(1 - x**2) / x**2, [x > 0, x < 1])
    assert _checks(x * sqrt(x**2 - 1), [x > 1]) and _checks(1 / sqrt(2 * x - x**2), [x > 0, x < 2])
    assert _checks(x**Rational(1, 3) / (x + 1), [x > 0]) and _checks(sqrt(x) / (x + 1), [x > 0])


def test_gr_2_3_exponential() -> None:
    # 2.311, 2.321, 2.322, 2.325
    assert _same(heurisch_antiderivative(exp(a * x), x), exp(a * x) / a)
    assert _same(heurisch_antiderivative(x * exp(a * x), x), exp(a * x) * (a * x - 1) / a**2)
    assert _same(heurisch_antiderivative(x**2 * exp(a * x), x), exp(a * x) * (a**2 * x**2 - 2 * a * x + 2) / a**3)
    assert _same(heurisch_antiderivative(1 / (exp(x) + 1), x), x - log(exp(x) + 1))
    assert _checks(exp(x) / (exp(2 * x) - 1)) and _checks(exp(x) * sin(x)) and _checks(exp(a * x) * cos(b * x))
    assert _checks(exp(-x**2) * x**3) and _checks(exp(sqrt(x)) / sqrt(x), [x > 0])


def test_gr_2_4_hyperbolic() -> None:
    assert _same(heurisch_antiderivative(sinh(x)**2, x), sinh(x) * cosh(x) / 2 - x / 2)
    assert _checks(cosh(x)**3) and _checks(1 / (cosh(x) + 1)) and _checks(sinh(x) / (cosh(x)**2 + 1))
    assert _checks(x * sinh(x)) and _checks(exp(x) * sinh(x))


def test_gr_2_5_trigonometric() -> None:
    # 2.513, 2.526, 2.553, 2.554
    assert _same(heurisch_antiderivative(sin(x)**2, x), x / 2 - sin(x) * cos(x) / 2)
    assert heurisch_antiderivative(1 / sin(x), x) == log(tan(x / 2))
    assert heurisch_antiderivative(1 / (1 + cos(x)), x) == tan(x / 2) and _checks(1 / (1 + cos(x)))
    # 2.553.3 with tan(x/2) = sin(x)/(1 + cos(x))
    assert heurisch_antiderivative(1 / (2 + cos(x)), x) == 2 * sqrt(3) * atan(sqrt(3) * tan(x / 2) / 3) / 3
    assert _checks(1 / (2 + cos(x)))
    assert _checks(sin(x)**4) and _checks(sin(x)**2 * cos(x)**3) and _checks(sin(a * x) * cos(b * x))
    assert _checks(x * sin(x)**2) and _checks(x**2 * cos(x)) and _checks(1 / (sin(x) + cos(x)))


def test_gr_2_7_logarithmic() -> None:
    # 2.721, 2.722, 2.727
    assert _same(heurisch_antiderivative(log(x)**2, x), x * log(x)**2 - 2 * x * log(x) + 2 * x)
    assert _same(heurisch_antiderivative(x**2 * log(x), x), x**3 * log(x) / 3 - x**3 / 9)
    assert _same(heurisch_antiderivative(1 / (x * log(x)**2), x), -1 / log(x))
    assert _checks(log(x)**3) and _checks(log(x) / x**3) and _checks(log(x + 1) / x**2) and _checks(x * log(x**2 + 1))
    assert _checks(log(x)**2 / x) and _checks(log(x)**2 / x**2)
    # log(log(x))/x needs log(x)*log(log(x)) - log(x), beyond the degree
    # bound of the polynomial part here as in SymPy
    assert heurisch_antiderivative(log(log(x)) / x, x) is None
