"""The typed zeta, dirichlet_eta and polylog against SymPy (the oracle),
compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import nan
from sympy_extras.core.power import sqrt
from sympy_extras.functions import dirichlet_eta, polylog, zeta
from sympy_extras.functions.special.zeta_functions import bernoulli_number

x, s = Symbol('x'), Symbol('s')
n = Symbol('n', integer=True)
p = Symbol('p', positive=True)


def orders() -> list[Expr]:
    return [Integer(-3), Integer(-2), Integer(-1), Integer(0), Integer(1), Integer(2), Integer(3), Integer(4),
            Integer(6), Rational(1, 2), oo, nan, s, n, p]


def points() -> list[Expr]:
    return [Integer(1), Integer(2), Integer(3), Integer(0), Integer(-1), Integer(-2), Rational(1, 2), Rational(1, 3),
            oo, nan, x, p, 2*x]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def test_zeta_and_eta_match_sympy() -> None:
    different = []
    for a in orders():
        sa = sympy_value(a)
        cases = [(zeta(a), sympy.zeta(sa)), (dirichlet_eta(a), sympy.dirichlet_eta(sa))]
        for b in points():
            sb = sympy_value(b)
            cases.append((zeta(a, b), sympy.zeta(sa, sb)))
            cases.append((dirichlet_eta(a, b), sympy.dirichlet_eta(sa, sb)))
        for ours, theirs in cases:
            if str(to_sympy(ours, evaluate=False)) != str(theirs):
                different.append((str(ours), str(theirs)))
    assert not different, different


def test_polylog_matches_sympy() -> None:
    five = sqrt(Integer(5))
    arguments: list[Expr] = points() + [I, -I, 1 + I, 1 - I, (1 - I)/2, (five - 1)/2, -(five - 1)/2,
                                        -(five + 1)/2, (3 - five)/2, Integer(2)]
    different = []
    for a, b in itertools.product(orders(), arguments):
        ours, theirs = str(to_sympy(polylog(a, b), evaluate=False)), str(sympy.polylog(sympy_value(a), sympy_value(b)))
        if ours != theirs:
            different.append((str(a), str(b), ours, theirs))
    assert not different, different


def test_derivatives_numbers_and_bernoulli() -> None:
    sx, ss = sympy.Symbol('x'), sympy.Symbol('s')
    for ours, theirs, v, sv in [(zeta(s, x), sympy.zeta(ss, sx), x, sx), (zeta(s, x), sympy.zeta(ss, sx), s, ss),
                                (polylog(s, x**2), sympy.polylog(ss, sx**2), x, sx), (zeta(s**2), sympy.zeta(ss**2), s, ss)]:
        assert str(to_sympy(ours.diff(v), evaluate=False)) == str(theirs.diff(sv)), (ours, v)
    for e, t in [(zeta(Integer(3)), sympy.zeta(3)), (dirichlet_eta(Integer(3)), sympy.dirichlet_eta(3)),
                 (polylog(Integer(3), Rational(1, 3)), sympy.polylog(3, sympy.Rational(1, 3))),
                 (zeta(Rational(3, 2), Integer(-2) + I), sympy.zeta(sympy.Rational(3, 2), -2 + sympy.I))]:
        value = complex(sympy_value(e.evalf(20)))
        assert abs(value - complex(sympy.N(t, 20))) < 1e-14, e
    for k in range(30):
        b = bernoulli_number(k)
        assert sympy.bernoulli(k) == sympy.Rational(b.numerator, b.denominator), k
