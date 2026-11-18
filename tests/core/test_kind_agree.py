"""``sympy_extras.core.kind`` against its Rust port: the kinds of the
nodes of every class (numbers, symbols commutative or not, sums,
products, powers, functions, applied undefined functions, derivatives,
substitutions, relations and Booleans, sets, tuples, piecewise
expressions, roots of polynomials, bounds)."""

from __future__ import annotations

from typing import Any

from ..conftest import load


def _nodes() -> list[Any]:
    from sympy_extras.core import (And, Derivative, Eq, Function, I, Integer, Lt, Not, Or, Rational,
                                   Subs, Symbol, Tuple, false, oo, pi, sqrt, true, zoo, nan, E)
    from sympy_extras.core.accumbounds import AccumBounds
    from sympy_extras.functions import Piecewise, exp, gamma, hyper, log, sin
    from sympy_extras.sets import FiniteSet, Interval, Union
    x, y = Symbol('x'), Symbol('y')
    A = Symbol('A', commutative=False)
    f = Function('f')
    return [Integer(2), Rational(1, 3), pi, E, I, oo, zoo, nan, x, A, x + y, x*A, A*A, x**2, sqrt(x), exp(x), log(x),
            sin(x), gamma(x), f(x), f(A), Derivative(f(x), x), Subs(f(x), x, 1), Derivative(sin(x)*f(x), x), Eq(x, 1),
            Lt(x, 2), And(Lt(x, 1), Eq(y, 2)), Or(Lt(x, 1), Not(Eq(y, 2))), true, false, Interval(0, 1),
            FiniteSet(1, 2), Union(Interval(0, 1), FiniteSet(3)), Tuple(x, Eq(x, 1)), Tuple(x), Tuple(), Tuple(Tuple(x), A),
            Piecewise((x, Lt(x, 0)), (1, True)), hyper([1], [2], x), AccumBounds(-1, 1), x + AccumBounds(-1, 1),
            exp(A), sin(x)**2 + exp(x)]


def test_kinds(rust: Any) -> None:
    rc = load("sympy_extras_rs.core")
    for node in _nodes():
        assert str(rc.logic_call('kind', [node])) == str(node.kind), node
