"""The typed Hankel, spherical Bessel and Airy functions against SymPy (the
oracle), compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools
from typing import Callable

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import nan
from sympy_extras.functions import (AiryAi, airyai, airyaiprime, airybi, airybiprime, conjugate, hankel1, hankel2, jn,
                                    yn)

x = Symbol('x')
r = Symbol('r', real=True)
p = Symbol('p', positive=True)
n = Symbol('n', integer=True)
nu = Symbol('nu')

BINARY: list[tuple[Callable[[Expr, Expr], Expr], Callable[[sympy.Expr, sympy.Expr], sympy.Expr]]] = [
    (hankel1, sympy.hankel1), (hankel2, sympy.hankel2), (jn, sympy.jn), (yn, sympy.yn)]
UNARY: list[tuple[Callable[[Expr], Expr], Callable[[sympy.Expr], sympy.Expr]]] = [
    (airyai, sympy.airyai), (airybi, sympy.airybi), (airyaiprime, sympy.airyaiprime), (airybiprime, sympy.airybiprime)]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_values_match_sympy() -> None:
    different: list[tuple[str, ...]] = []
    orders: list[Expr] = [Integer(0), Integer(1), Integer(-2), Rational(1, 2), n, nu, p]
    points: list[Expr] = [Integer(0), Integer(2), oo, -oo, x, -x, I*x, p, r]
    for order, z in itertools.product(orders, points):
        for f, g in BINARY:
            if not same(f(order, z), g(sympy_value(order), sympy_value(z))):
                different.append((f.__name__, str(order), str(z)))
    for z in points + [nan, I]:
        for f1, g1 in UNARY:
            if not same(f1(z), g1(sympy_value(z))):
                different.append((f1.__name__, str(z), str(f1(z)), str(g1(sympy_value(z)))))
    assert not different, different


def test_derivatives_conjugates_parts_numbers() -> None:
    sx, snu = sympy.Symbol('x'), sympy.Symbol('nu')
    for f, g in BINARY:
        assert same(f(nu, x**2).diff(x), g(snu, sx**2).diff(sx)), f
        assert same(conjugate(f(nu, x)), sympy.conjugate(g(snu, sx))), f
        cases: list[tuple[Expr, Expr]] = [(Integer(1), Rational(1, 3)), (Integer(2), Integer(3))]
        for order, value in cases:
            ours = complex(sympy_value(f(order, x).subs(x, value).evalf(20)))
            assert abs(ours - complex(sympy.N(g(sympy_value(order), sympy_value(value)), 20))) < 1e-13, (f, value)
    for f1, g1 in UNARY:
        assert same(f1(x**2).diff(x), g1(sx**2).diff(sx)), f1
        ours_parts = tuple(str(to_sympy(t, evaluate=False)) for t in f1(x).as_real_imag())
        assert ours_parts == tuple(str(t) for t in g1(sx).as_real_imag()), f1
        points: list[Expr] = [Rational(1, 3), Integer(-2), 1 + I]
        for point in points:
            ours = complex(sympy_value(f1(x).subs(x, point).evalf(20)))
            assert abs(ours - complex(sympy.N(g1(sympy_value(point)), 20))) < 1e-13, (f1, point)
    assert AiryAi(r).is_extended_real is True
