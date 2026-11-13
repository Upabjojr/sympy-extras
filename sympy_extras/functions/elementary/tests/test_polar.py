"""The typed polar numbers (exp_polar, polar_lift, periodic_argument,
principal_branch, polarify, unpolarify) against SymPy (the oracle),
compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.functions import (abs_, exp, exp_polar, periodic_argument, polar_lift, polarify, principal_branch,
                                    unbranched_argument, unpolarify)

x, y = Symbol('x'), Symbol('y')
p = Symbol('p', positive=True)
q = Symbol('q', polar=True)


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def values() -> list[Expr]:
    return [Integer(3), Integer(-4), I, -I, 1 + I, x, -x, p, x*p, q, 2*q, exp_polar(I*pi), exp_polar(2*I*pi)*3,
            exp_polar(10*I*pi), exp_polar(15*I), exp_polar(x), polar_lift(x), exp_polar(I*pi/3)*p, q**2,
            exp_polar(I*pi)**Rational(1, 2), x*exp_polar(3*I*pi)]


def test_values_match_sympy() -> None:
    different = []
    for a in values():
        sa = sympy_value(a)
        cases: list[tuple[Expr, sympy.Basic]] = [(polar_lift(a), sympy.polar_lift(sa)), (unpolarify(a), sympy.unpolarify(sa)),
                                                 (unbranched_argument(a), sympy.unbranched_argument(sa)),
                                                 (abs_(a), sympy.Abs(sa))]
        for period in [2*pi, oo, 4*pi]:
            sp = sympy_value(period)
            cases.append((periodic_argument(a, period), sympy.periodic_argument(sa, sp)))
            cases.append((principal_branch(a, period), sympy.principal_branch(sa, sp)))
        for ours, theirs in cases:
            if not same(ours, theirs):
                different.append((str(ours), str(theirs)))
    assert not different, different[:12]


def test_products_powers_and_polarify() -> None:
    exponents: list[Expr] = [Integer(1), Integer(2), x, -x, 2*I*pi, -2*I*pi, I, Integer(0)]
    for u, v in itertools.product(exponents, repeat=2):
        assert same(exp_polar(u)*exp_polar(v), sympy.exp_polar(sympy_value(u))*sympy.exp_polar(sympy_value(v))), (u, v)
        assert same(exp_polar(u)**v, sympy.exp_polar(sympy_value(u))**sympy_value(v)), (u, v)
    expressions: list[Expr] = [2*x, x*y, x + 1, exp(x), Integer(3)]
    for e in expressions:
        ours, _ = polarify(e, subs=False)
        assert same(ours, sympy.polarify(sympy_value(e), subs=False)), e
        lifted, _ = polarify(e, lift=True)
        assert same(lifted, sympy.polarify(sympy_value(e), lift=True)), e
    ours_e, reps = polarify(x*y)
    theirs_e, theirs_reps = sympy.polarify(sympy.Symbol('x')*sympy.Symbol('y'))
    assert str(ours_e) == str(theirs_e) and sorted(map(str, reps.values())) == sorted(map(str, theirs_reps.values()))
    assert all(r.is_polar for r in reps)
    assert str(exp_polar(I*pi).evalf()) == str(sympy.exp_polar(sympy.I*sympy.pi).evalf())
    # SymPy's exp_polar has no derivative rule: Derivative(exp_polar(x), x)
    from sympy_extras.core.function import Derivative
    assert exp_polar(x).is_polar is True and exp_polar(x).diff(x) == Derivative(exp_polar(x), x)



def test_exp_polar_of_a_real_number_is_positive() -> None:
    # exp_polar(0).is_positive was None (SymPy's is True, found numerically):
    # the conditions of the Meijer G-function theorems ask whether the
    # polar coefficient of an argument, x*exp_polar(0), is positive
    assert exp_polar(Integer(0)).is_positive is True
    assert exp_polar(Integer(2)).is_positive is True
    assert exp_polar(I * pi).is_positive is not True
    assert exp_polar(x).is_positive is None
