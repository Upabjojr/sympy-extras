"""Tests of the native Hermite reduction and reduction-based telescopers on
the objects of the core (:mod:`sympy_extras.integrals.reduction`)."""
from __future__ import annotations

from sympy_extras.core import Integer, Rational, Symbol, diff, oo
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, exp
from sympy_extras.integrals.reduction import (HermiteReducer, hermite_reduce, reduction_integral, reduction_ode,
                                              reduction_telescoper)
from sympy_extras.integrals.telescoping import almkvist_zeilberger
from sympy_extras.polys.polytools import cancel
from sympy_extras.settings import configure

x = Symbol('x')
t = Symbol('t')
s = Symbol('s', positive=True)


def _reduces(p: Expr, q: Expr) -> tuple[Expr, Expr]:
    """``p = g' + q g + r``, checked."""
    g, r = hermite_reduce(p, q, x, t)
    assert cancel(diff(g, x) + q*g + r - p) == 0, (p, q)
    return g, r


def test_reduction_identity() -> None:
    for p, q in [(1/x**2, -2*x), (x**3, -2*x), (1/x**3, Integer(-1)), (1/(x**2 + 1)**2, Integer(0)),
                 (x**2/(x - 1)**2, -2*x + t), (1/((x - t)*(x**2 + 1)**2), t/x**2), (x**4, -5*x/(x**2 + t))]:
        _reduces(p, q)


def test_integrable_functions_reduce_to_zero() -> None:
    # x exp(-x**2) = (-exp(-x**2)/2)'
    assert _reduces(x, -2*x) == (Rational(-1, 2), 0)
    # D(g) reduces to zero for any g
    q = -2*x + t/x
    g = (x**2 + 1)/(x - 3)**2
    p = cancel(diff(g, x) + q*g)
    assert _reduces(p, q)[1] == 0
    # exp(-x**2) is not integrable
    assert _reduces(Integer(1), -2*x) == (0, 1)


def test_shell_and_conjugate_poles() -> None:
    reducer = HermiteReducer(2/x + 1/(x - t)**2, x, t)
    assert reducer.shell == x**2 and reducer.A == 1 and cancel(reducer.B - (x - t)**2) == 0
    g, r = reducer.reduce(1/x)
    assert g == t**2/2 - t*x + x**2/2 and r == t - Rational(1, 2)
    # conjugate poles with non-integer residues 1 -+ I: factored over QQ_I
    assert _reduces(1/(x**2 + 1)**2, (2*x + 2)/(x**2 + 1)) == (1/(2*x**2 + 2), 0)


def test_reduced_forms_are_linear() -> None:
    reducer = HermiteReducer(-2*x + t, x, t)
    _, r1 = reducer.reduce(x**2/(x - 1)**2)
    _, r2 = reducer.reduce(3*x**2/(x - 1)**2 + x**4)
    _, r3 = reducer.reduce(x**4)
    assert cancel(r2 - 3*r1 - r3) == 0


def test_a_new_parameter_extends_the_field() -> None:
    a = Symbol('a')
    reducer = HermiteReducer(-2*x, x, t)
    g, r = reducer.reduce(a*x)
    assert (g, r) == (-a/2, 0)


def test_telescopers_agree_with_the_ansatz() -> None:
    for F in [exp(-x**2 - s**2/x**2), 1/(x**2 + s**2), exp(-x**2)*exp(2*x*s), exp(-x**2)*cos(2*s*x)]:
        found = reduction_telescoper(F, x, s)
        other = almkvist_zeilberger(F, x, s)
        assert found is not None and other is not None and found.check(), F
        assert found.order <= other.order, F
    found = reduction_telescoper(exp(-s*x**2) + exp(-s**2*x**2), x, s)
    assert found is not None and found.coefficients == [1, 5*s, 2*s**2] and found.check()


def test_equations_and_integrals() -> None:
    equation = reduction_ode(exp(-x**2)*exp(2*x*s), x, -oo, oo, s)
    assert equation is not None and equation.rhs == 0
    with configure(numerical_checks=False):
        found = reduction_integral(x*exp(-s*x**2), x, 0, oo, s)
    assert found is not None and found.value == 1/(2*s)
