"""Tests of differentiation under the integral sign on the objects of the
core (:mod:`sympy_extras.integrals.parametric`), with the classical values
of Gradshteyn–Ryzhik 3.941.1, 4.295.7 and Woods' textbook example, and a
custom :class:`ParametricIntegrator`."""
from __future__ import annotations

from typing import Optional

from sympy_extras.core import S, Symbol, oo, pi
from sympy_extras.core.expr import Expr
from sympy_extras.functions import atan, exp, log, sin
from sympy_extras.integrals.conditions import ConditionalValue, numerically_equal
from sympy_extras.integrals.parametric import ParametricIntegrator, candidate_parameters, parametric_integral

x = Symbol('x')
p, t = Symbol('p', positive=True), Symbol('t', positive=True)


def test_candidate_parameters() -> None:
    assert candidate_parameters(exp(-p * x) * sin(x) / x, x) == [p]
    assert candidate_parameters(t * exp(-x), x) == []
    assert candidate_parameters(x**p * t**x, x) == [p, t]


def test_classical_values() -> None:
    # GR 3.941.1: acot(p)
    found = parametric_integral(exp(-p * x) * sin(x) / x, x, S.Zero, oo, p)
    assert found is not None and numerically_equal(found.value, pi / 2 - atan(p))
    # Woods: Integral((x^p - 1)/log(x), (x, 0, 1)) = log(1 + p)
    found = parametric_integral((x**p - 1) / log(x), x, S.Zero, S.One, p)
    assert found is not None and numerically_equal(found.value, log(1 + p))


def test_not_applicable() -> None:
    assert parametric_integral(exp(-x), x, S.Zero, oo, p) is None


class _Table(ParametricIntegrator):
    """An integrator which knows two integrals over (0, 1) and checks
    nothing: the boundary is a plain typed class."""

    def __init__(self) -> None:
        self.calls: list[Expr] = []

    def integrate(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        self.calls.append(f)
        if f == x**p:
            return ConditionalValue(1 / (p + 1))
        if f == S.Zero:
            return ConditionalValue(S.Zero)
        return None

    def verify(self, value: Expr, f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[bool]:
        return None


def test_a_custom_integrator() -> None:
    # d/dp (x**p - 1)/log(x) = x**p, whose integral the table gives; the
    # anchor p = 0 is the integral of 0
    integrator = _Table()
    found = parametric_integral((x**p - 1) / log(x), x, S.Zero, S.One, p, None, 0, integrator)
    assert found is not None and numerically_equal(found.value, log(p + 1))
    assert integrator.calls[0] == x**p
