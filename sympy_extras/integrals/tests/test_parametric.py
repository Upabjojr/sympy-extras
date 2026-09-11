"""Tests of differentiation under the integral sign (with the classical
values: Gradshteyn–Ryzhik 3.941.1, 4.295.7, 4.267.8 and Woods' textbook
example)."""
from __future__ import annotations

from sympy import symbols, exp, sin, log, oo, atan, pi, simplify, cos, sqrt, S

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals import definite_integral
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.parametric import parametric_integral, candidate_parameters

x = symbols('x')
p, t = symbols('p t', positive=True)


def _same(u: ExprLike, v: ExprLike) -> bool:
    return simplify(as_expr(u) - as_expr(v)) == 0


def test_candidate_parameters() -> None:
    assert candidate_parameters(exp(-p * x) * sin(x) / x, x) == [p]
    # a parameter which only multiplies the integrand is not a candidate
    assert candidate_parameters(t * exp(-x), x) == []
    assert candidate_parameters(log(1 + t**2 * x**2) / (1 + x**2), x) == [t]


def test_feynman_examples() -> None:
    # GR 3.941.1: Integral(exp(-p x) sin(x)/x) = acot(p); the derivative in p is -1/(1 + p^2)
    found = parametric_integral(exp(-p * x) * sin(x) / x, x, S.Zero, oo, p)
    assert found is not None and _same(found.value, pi / 2 - atan(p))
    # GR 4.295.7: Integral(log(1 + t^2 x^2)/(1 + x^2), (x, 0, oo)) = pi log(1 + t)
    found = parametric_integral(log(1 + t**2 * x**2) / (1 + x**2), x, S.Zero, oo, t)
    assert found is not None and _same(found.value, pi * log(1 + t))
    # Woods: Integral((x^p - 1)/log(x), (x, 0, 1)) = log(1 + p)
    found = parametric_integral((x**p - 1) / log(x), x, S.Zero, S.One, p)
    assert found is not None and _same(found.value, log(1 + p))
    # GR 4.535.1: Integral(atan(p x)/(x (1 + x^2)), (x, 0, oo)) = pi/2 log(1 + p)
    found = parametric_integral(atan(p * x) / (x * (1 + x**2)), x, S.Zero, oo, p)
    assert found is not None and _same(found.value, pi * log(1 + p) / 2)
    for value, f in [(pi * log(1 + p) / 2, atan(p * x) / (x * (1 + x**2))), (log(1 + p), (x**p - 1) / log(x))]:
        assert verify_numerically(value, f, x, S.Zero, oo if f.has(atan) else S.One) is not False


def test_not_applicable() -> None:
    # no parameter inside a function of x, or the derivative no easier
    assert parametric_integral(exp(-x), x, S.Zero, oo, p) is None
    assert parametric_integral(exp(-x * p**2) * sqrt(cos(x) + 2), x, S.Zero, oo, p) is None


def test_through_the_driver() -> None:
    assert _same(definite_integral(atan(p * x) / (x * (1 + x**2)), (x, 0, oo)), pi * log(1 + p) / 2)
    assert _same(definite_integral((x**p - 1) / log(x), (x, 0, 1)), log(1 + p))
