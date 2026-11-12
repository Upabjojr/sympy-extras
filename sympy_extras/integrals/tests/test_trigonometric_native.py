"""Tests of the native trigonometric and hyperbolic integrator
(:mod:`sympy_extras.integrals.trigonometric`) on the objects of the core,
against the closed forms of Gradshteyn and Ryzhik and by differentiation."""
from __future__ import annotations

from typing import Optional

from sympy_extras.core import Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean
from sympy_extras.core.relational import ne
from sympy_extras.functions import (ExprCondPair, Piecewise, acos, asin, asinh, atan, cos, cosh, cot, csc, exp, log,
                                    sec, sin, sinh, tan, tanh)
from sympy_extras.integrals.conditions import numerically_equal
from sympy_extras.integrals.trigonometric import (_sin_cos, _sinh_cosh, _solutions, _verified,
                                                  trigonometric_antiderivative)

x, u = Symbol('x'), Symbol('u')
a, b = Symbol('a', positive=True), Symbol('b', positive=True)
_INSIDE: list[Boolean] = [x > Rational(1, 10), x < Rational(1, 2)]


def _checks(f: Expr, facts: Optional[list[Boolean]] = None) -> bool:
    F = trigonometric_antiderivative(f, x)
    if F is None:
        return False
    if isinstance(F, Piecewise):
        pair = F.args[0]
        assert isinstance(pair, ExprCondPair)
        F = pair.expr
    return numerically_equal(F.diff(x), f, _INSIDE if facts is None else facts)


def test_closed_forms() -> None:
    # GR 2.513.1, 2.526.2 (sec), 2.526.1 (csc), 2.423 (tanh**3)
    # (factor_terms keeps the half outside the sum, as on SymPy objects)
    assert str(trigonometric_antiderivative(sin(x)**2, x)) == '(x - sin(x)*cos(x))/2'
    assert trigonometric_antiderivative(sin(x)**3 * cos(x)**2, x) == cos(x)**5 / 5 - cos(x)**3 / 3
    assert trigonometric_antiderivative(tan(x)**3, x) == log(cos(x)) + tan(x)**2 / 2
    assert trigonometric_antiderivative(sec(x), x) == log(tan(x) + sec(x))
    assert trigonometric_antiderivative(csc(x), x) == log(tan(x / 2))
    assert trigonometric_antiderivative(sin(x)**(-2), x) == -cot(x)
    assert trigonometric_antiderivative(asin(x), x) == x * asin(x) + (1 - x**2)**Rational(1, 2)
    assert trigonometric_antiderivative(tanh(x)**3, x) is not None


def test_powers_of_one_argument() -> None:
    for n in range(-3, 4):
        for m in range(-3, 4):
            if (n, m) != (0, 0):
                assert _checks(sin(x)**n * cos(x)**m), (n, m)
                assert _checks(sinh(x)**n * cosh(x)**m, [x > 0]), (n, m)
    for n, m in [(-2, -2), (-4, 2), (2, -4), (4, 4)]:
        F = _sin_cos(n, m, u)
        assert F is not None and numerically_equal(F.diff(u), sin(u)**n * cos(u)**m, [u > 0, u < 1]), (n, m)
        G = _sinh_cosh(n, m, u)
        assert G is not None and numerically_equal(G.diff(u), sinh(u)**n * cosh(u)**m, [u > 0, u < 1]), (n, m)


def test_polynomials_exponentials_and_products() -> None:
    for f in [x * exp(x) * sin(x), x**2 * exp(-x) * cos(x), x**3 * sin(a * x), x * sinh(2 * x),
              x * sin(x)**2, exp(2 * x) * sin(3 * x) * cos(x), exp(-x) * sinh(x)**2, sin(2 * x) * sin(5 * x) * cos(x)]:
        assert _checks(f), f
    # the frequency a - b may vanish: the case a = b apart
    F = trigonometric_antiderivative(sin(a * x) * cos(b * x), x)
    assert isinstance(F, Piecewise)
    pair = F.args[0]
    assert isinstance(pair, ExprCondPair) and pair.cond == ne(a - b, 0)


def test_solutions_respect_the_assumptions() -> None:
    # SymPy's solve(a + b, b) with b > 0 has no solution: the native
    # replacement leaves out the roots which contradict the assumptions
    assert _solutions(a - b, b) == [a]
    assert _solutions(a + b, b) == []
    assert _solutions(sin(b), b) is None


def test_rational_functions_and_inverse_functions() -> None:
    # the Weierstrass substitution and the integrations by parts leave
    # rational functions, integrated by SymPy's ratint at run time
    for f in [1 / (2 + cos(x)), sin(x) / (1 + cos(x)**2), 1 / (1 + tan(x)), 1 / (sinh(x) + 2), x**2 * atan(x),
              x**3 * acos(x), x * asinh(2 * x), acos(x)**2]:
        assert _checks(f), f
    assert trigonometric_antiderivative(atan(x)**2, x) is None
    assert trigonometric_antiderivative(x**2, x) is None
    assert trigonometric_antiderivative(sin(x**2), x) is None


def test_every_result_is_verified() -> None:
    assert _verified(-cos(x), sin(x), x) and not _verified(cos(x), sin(x), x)
    assert not _verified(x * asin(x), asin(x), x)
