"""The typed Bessel functions against SymPy (the oracle), compared
without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools
from typing import Callable

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import nan
from sympy_extras.functions import BesselI, BesselJ, BesselK, BesselY, besseli, besselj, besselk, bessely

x = Symbol('x')
r = Symbol('r', real=True)
p = Symbol('p', positive=True)
n = Symbol('n', integer=True)
nu = Symbol('nu')

FUNCTIONS: list[tuple[Callable[[Expr, Expr], Expr], Callable[[sympy.Expr, sympy.Expr], sympy.Expr]]] = [
    (besselj, sympy.besselj), (bessely, sympy.bessely), (besseli, sympy.besseli), (besselk, sympy.besselk)]


def orders() -> list[Expr]:
    return [Integer(0), Integer(1), Integer(-2), Rational(1, 2), Rational(-3, 2), I, 1 + I, n, -n, nu, p, -p, nan]


def points() -> list[Expr]:
    return [Integer(0), Integer(2), oo, -oo, I*oo, -I*oo, x, -x, I*x, -I*x, 2*I*p, p, -p, r]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def test_values_match_sympy() -> None:
    different = []
    for order, z in itertools.product(orders(), points()):
        so, sz = sympy_value(order), sympy_value(z)
        for ours_f, theirs_f in FUNCTIONS:
            ours, theirs = str(to_sympy(ours_f(order, z), evaluate=False)), str(theirs_f(so, sz))
            if ours != theirs:
                different.append((ours_f.__name__, str(order), str(z), ours, theirs))
    # besseli(nu, -oo) = (-1)**nu*oo: the core decides the sign of (-1)**I =
    # exp(-pi) and (-1)**(1 + I) = -exp(-pi), SymPy leaves the product
    accepted = {('besseli', 'I', '-oo'), ('besseli', '1 + I', '-oo')}
    unexpected = [d for d in different if (d[0], d[1], d[2]) not in accepted]
    assert not unexpected, unexpected[:20]


def test_facts_derivatives_numbers() -> None:
    for order, z in itertools.product(orders(), points()):
        for cls, g in [(BesselJ, sympy.besselj), (BesselY, sympy.bessely), (BesselI, sympy.besseli),
                       (BesselK, sympy.besselk)]:
            got = cls(order, z)._ask('extended_real')
            if got is not None:
                assert got == g(sympy_value(order), sympy_value(z), evaluate=False).is_extended_real, (cls, order, z)
    sx, snu = sympy.Symbol('x'), sympy.Symbol('nu')
    for f, g in FUNCTIONS:
        assert str(to_sympy(f(nu, x**2).diff(x), evaluate=False)) == str(g(snu, sx**2).diff(sx)), f
        for order, value in [(Integer(1), Rational(1, 3)), (Rational(1, 2), Integer(3)), (Integer(2), 2 + I)]:
            ours = complex(sympy_value(f(order, x).subs(x, value).evalf(20)))
            assert abs(ours - complex(sympy.N(g(sympy_value(order), sympy_value(value)), 20))) < 1e-13, (f, value)
