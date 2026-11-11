"""The typed lowergamma, uppergamma, expint and Ei against SymPy (the
oracle), compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import nan
from sympy_extras.functions import Ei, expint, lowergamma, uppergamma

x = Symbol('x')
p = Symbol('p', positive=True)
n = Symbol('n', integer=True)
a = Symbol('a')


def orders() -> list[Expr]:
    return [Integer(0), Integer(1), Integer(2), Integer(4), Integer(-1), Integer(-3), Rational(1, 2), Rational(5, 2),
            Rational(-1, 2), Rational(-5, 2), Rational(1, 3), n, a, p, I]


def points() -> list[Expr]:
    return [Integer(0), Integer(1), Integer(3), oo, -oo, nan, x, p, -p, 2*x, I]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def test_values_match_sympy() -> None:
    different = []
    for nu, z in itertools.product(orders(), points()):
        snu, sz = sympy_value(nu), sympy_value(z)
        for ours, theirs in [(lowergamma(nu, z), sympy.lowergamma(snu, sz)),
                             (uppergamma(nu, z), sympy.uppergamma(snu, sz)), (expint(nu, z), sympy.expint(snu, sz))]:
            if str(to_sympy(ours, evaluate=False)) != str(theirs):
                different.append((str(ours), str(theirs)))
    for z in points():
        if str(to_sympy(Ei(z), evaluate=False)) != str(sympy.Ei(sympy_value(z))):
            different.append((str(Ei(z)), str(sympy.Ei(sympy_value(z)))))
    assert not different, different[:20]


def test_derivatives_and_numbers() -> None:
    sx, sa = sympy.Symbol('x'), sympy.Symbol('a')
    for ours, theirs in [(lowergamma(a, x**2), sympy.lowergamma(sa, sx**2)), (uppergamma(a, x), sympy.uppergamma(sa, sx)),
                         (expint(a, x), sympy.expint(sa, sx)), (Ei(x**2), sympy.Ei(sx**2))]:
        assert str(to_sympy(ours.diff(x), evaluate=False)) == str(theirs.diff(sx)), ours
    for e, s in [(lowergamma(Rational(1, 3), x).subs(x, Integer(2)), sympy.lowergamma(sympy.Rational(1, 3), 2)),
                 (uppergamma(Rational(1, 3), x).subs(x, Integer(2)), sympy.uppergamma(sympy.Rational(1, 3), 2)),
                 (expint(Rational(1, 3), x).subs(x, Integer(2)), sympy.expint(sympy.Rational(1, 3), 2)),
                 (Ei(x).subs(x, Integer(-2)), sympy.Ei(-2)), (Ei(x).subs(x, 1 + I), sympy.Ei(1 + sympy.I))]:
        assert abs(complex(sympy_value(e.evalf(20))) - complex(sympy.N(s, 20))) < 1e-14, e


def test_branch_factors_of_polar_arguments_match_sympy() -> None:
    import sympy
    from sympy import I as sI, Rational as R, exp_polar as E, pi as sP
    from sympy_extras.interop import to_sympy
    from sympy_extras.interop import expr_from_sympy
    from sympy_extras.functions import lowergamma, uppergamma
    z, a = sympy.symbols('z a')
    p = sympy.Symbol('p', positive=True)
    points = [z, z*E(sI*sP), z*E(2*sI*sP), z*E(-2*sI*sP), z*E(4*sI*sP), p*E(sI*sP/2), z*E(3*sI*sP), 2*E(2*sI*sP)]
    for ours, theirs in [(lowergamma, sympy.lowergamma), (uppergamma, sympy.uppergamma)]:
        for s in [R(1), R(2), R(1, 2), R(1, 3), R(0), R(-1), R(-2), R(-3, 2), a]:
            for x in points:
                got = str(to_sympy(ours(expr_from_sympy(s), expr_from_sympy(x)), evaluate=False))
                assert got == str(theirs(s, x)), (theirs.__name__, s, x)
