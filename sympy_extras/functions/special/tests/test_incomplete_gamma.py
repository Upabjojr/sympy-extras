"""The typed lowergamma, uppergamma, expint and Ei against SymPy (the
oracle), compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, to_sympy
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
