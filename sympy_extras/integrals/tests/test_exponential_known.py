"""The antiderivatives of the exponential families against Mathematica's
``Integrate`` (Mathematica 14, ``Gamma[s, z]`` the upper incomplete gamma
function): the two antiderivatives differ by a constant, checked as the
equality of their increments between two points."""
from __future__ import annotations

from sympy import symbols, exp, sqrt, pi, erf, erfi, uppergamma, atanh, Rational, N
from sympy.core.expr import Expr

from sympy_extras._typing import as_expr
from sympy_extras.integrals.exponential import exponential_antiderivative

x = symbols('x')
v = symbols('v', positive=True)


def _same_increments(F: object, G: object, points: tuple[float, float], values: dict[Expr, float]) -> bool:
    fixed: dict[Expr, Expr] = {s: as_expr(Rational(str(value))) for s, value in values.items()}
    F_, G_ = as_expr(as_expr(F).xreplace(fixed)), as_expr(as_expr(G).xreplace(fixed))
    lower, upper = as_expr(Rational(str(points[0]))), as_expr(Rational(str(points[1])))
    left = as_expr(F_.subs(x, upper) - F_.subs(x, lower))
    right = as_expr(G_.subs(x, upper) - G_.subs(x, lower))
    return abs(complex(N(left - right, 20))) < 1e-12 * (1 + abs(complex(N(right, 20))))


def test_against_mathematica() -> None:
    a = symbols('a', negative=True)
    cases = [
        # Integrate[z^(v - 1) Exp[a z^2], z] with a < 0
        (x**(v - 1) * exp(a * x**2), -x**v * uppergamma(v / 2, -a * x**2) / (2 * (-a * x**2)**(v / 2)), [x > 0],
         {v: 1.7, a: -2.0}, (0.5, 1.5)),
        (sqrt(x) * exp(-2 * x**2), -(x**2)**Rational(1, 4) * uppergamma(Rational(3, 4), 2 * x**2) / (2 * 2**Rational(3, 4) * sqrt(x)),
         [x > 0], {}, (0.5, 1.5)),
        (x**2 * exp(x**2), exp(x**2) * x / 2 - sqrt(pi) * erfi(x) / 4, None, {}, (-1.0, 0.7)),
        (exp(x**3), -x * uppergamma(Rational(1, 3), -x**3) / (3 * (-x**3)**Rational(1, 3)), [x > 0], {}, (0.3, 1.1)),
        (x**5 * exp(-x**3), -(1 + x**3) * exp(-x**3) / 3, None, {}, (-1.0, 1.2)),
        (x**(-Rational(2, 3)) * exp(-x), -uppergamma(Rational(1, 3), x), [x > 0], {}, (0.4, 2.0)),
        (sqrt(x) * exp(-x), -uppergamma(Rational(3, 2), x), [x > 0], {}, (0.4, 2.0)),
        ((exp(x) + 2)**Rational(3, 2) / (exp(x) + 1),
         2 * sqrt(2 + exp(x)) + 2 * atanh(sqrt(2 + exp(x))) - 4 * sqrt(2) * atanh(sqrt(2 + exp(x)) / sqrt(2)), None, {}, (0.2, 1.4)),
        (exp(-x**3) / x**2, -(x**3)**Rational(1, 3) * uppergamma(-Rational(1, 3), x**3) / (3 * x), [x > 0], {}, (0.5, 1.3)),
        (x * exp(x**2 + 2 * x), (exp((1 + x)**2) - sqrt(pi) * erfi(1 + x)) / (2 * exp(1)), None, {}, (-0.5, 0.8)),
        (x**2 * exp(-x**2 + 3 * x), -(3 + 2 * x) * exp(-(x - 3) * x) / 4 - 11 * exp(Rational(9, 4)) * sqrt(pi) * erf(Rational(3, 2) - x) / 8,
         None, {}, (-0.5, 1.5)),
        (exp(2 / x) / x**3, exp(2 / x) * (x - 2) / (4 * x), [x > 0], {}, (0.7, 2.0)),
        (exp(sqrt(x)), 2 * exp(sqrt(x)) * (sqrt(x) - 1), [x > 0], {}, (0.5, 2.5)),
        (x**(v - 1) * exp(-x), -uppergamma(v, x), [x > 0], {v: 1.3}, (0.5, 2.0)),
    ]
    for f, mathematica, facts, values, points in cases:
        F = exponential_antiderivative(f, x, facts)
        assert F is not None, f
        assert _same_increments(F, mathematica, points, values), (f, F)
