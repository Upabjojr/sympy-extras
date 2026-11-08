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


def test_polar_arguments_match_sympy() -> None:
    # the branch rules of the Bessel functions, the trigonometric
    # integrals, Ei and expint on polar arguments
    import sympy
    from sympy import I as sI, Rational as R, exp_polar as E, pi as sP
    from sympy_extras.core import to_sympy
    from sympy_extras.core.interop import expr_from_sympy
    import sympy_extras.functions as F
    z, a = sympy.symbols('z a')
    p = sympy.Symbol('p', positive=True)
    args = [2*E(sI*sP/2), 2*E(sI*sP), z*E(sI*sP/2), z*E(2*sI*sP), z*E(-sI*sP/2), z*E(sI*sP), z*E(3*sI*sP), 2*sI, -z,
            p*E(-2*sI*sP), z*sympy.polar_lift(-1)]

    def ours(name: str, *xs: sympy.Expr) -> str:
        f = getattr(F, name)
        return str(to_sympy(f(*[expr_from_sympy(sympy.sympify(x)) for x in xs]), evaluate=False))
    for name in ['besselj', 'besseli', 'bessely', 'besselk']:
        for nu in [1, 0, R(1, 2), a, -1, 2]:
            for x in args:
                assert ours(name, nu, x) == str(getattr(sympy, name)(nu, x)), (name, nu, x)
    for name in ['Si', 'Ci', 'Shi', 'Chi', 'Ei']:
        for x in args:
            assert ours(name, x) == str(getattr(sympy, name)(x)), (name, x)
    for nu in [1, 2, 0, -1, R(1, 2), R(1, 3), a]:
        for x in args:
            assert ours('expint', nu, x) == str(sympy.expint(nu, x)), (nu, x)
