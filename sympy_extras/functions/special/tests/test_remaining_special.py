"""The typed erf2, erf2inv, betainc, lerchphi, riemann_xi, stieltjes and
multigamma against SymPy (the oracle), compared without letting SymPy
re-evaluate the results."""
from __future__ import annotations

import itertools

import pytest
import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, zoo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import nan
from sympy_extras.functions import (betainc, betainc_regularized, conjugate, erf2, erf2inv, erfinv, lerchphi, multigamma,
                                    riemann_xi, stieltjes)

x, y, z, s, a, b = (Symbol(c) for c in 'xyzsab')
r = Symbol('r', real=True)
n = Symbol('n', integer=True)
p = Symbol('p', integer=True, positive=True)


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_erf2_erf2inv_match_sympy() -> None:
    different = []
    values: list[Expr] = [Integer(0), Integer(1), oo, -oo, nan, x, -x, y, -y, r, I]
    for u, v in itertools.product(values, repeat=2):
        su, sv = sympy_value(u), sympy_value(v)
        for ours, theirs in [(erf2(u, v), sympy.erf2(su, sv)), (erf2inv(u, v), sympy.erf2inv(su, sv))]:
            if not same(ours, theirs):
                different.append((str(ours), str(theirs)))
    assert same(erf2(x, erf2inv(x, y)), sympy.erf2(sympy.Symbol('x'), sympy.erf2inv(sympy.Symbol('x'), sympy.Symbol('y'))))
    assert not different, different[:10]
    sx, sy = sympy.symbols('x y')
    for v, sv in [(x, sx), (y, sy)]:
        assert same(erf2(x, y).diff(v), sympy.erf2(sx, sy).diff(sv))
        assert same(erf2inv(x, y).diff(v), sympy.erf2inv(sx, sy).diff(sv))
    assert erfinv(Integer(0)) == 0


def test_betainc_lerchphi_match_sympy() -> None:
    sa, sb, sx, sy, sz, ss = sympy.symbols('a b x y z s')
    for f, g in [(betainc, sympy.betainc), (betainc_regularized, sympy.betainc_regularized)]:
        e, t = f(a, b, x, y), g(sa, sb, sx, sy)
        assert same(e, t)
        for v, sv in [(x, sx), (y, sy)]:
            assert same(e.diff(v), t.diff(sv)), (f, v)
        assert same(conjugate(e), sympy.conjugate(t))
        value = complex(sympy_value(f(Integer(2), Integer(3), Rational(1, 5), Rational(1, 2)).evalf(20)))
        assert abs(value - complex(sympy.N(g(2, 3, sympy.Rational(1, 5), sympy.Rational(1, 2)), 20))) < 1e-14
    e, t = lerchphi(z, s, a), sympy.lerchphi(sz, ss, sa)
    assert same(e, t)
    for v, sv in [(z, sz), (a, sa)]:
        assert same(e.diff(v), t.diff(sv)), v


def test_riemann_xi_stieltjes_multigamma_match_sympy() -> None:
    different = []
    for value in [Integer(k) for k in range(-4, 7)] + [s, Rational(1, 2), nan]:
        sv = sympy_value(value)
        if not same(riemann_xi(value), sympy.riemann_xi(sv)):
            different.append(('xi', str(riemann_xi(value)), str(sympy.riemann_xi(sv))))
        for second in [None, Integer(1), Integer(2), Integer(0), Integer(-1), nan, a]:
            ours = stieltjes(value) if second is None else stieltjes(value, second)
            theirs = sympy.stieltjes(sv) if second is None else sympy.stieltjes(sv, sympy_value(second))
            if not same(ours, theirs):
                different.append(('stieltjes', str(ours), str(theirs)))
    for order in [Integer(1), Integer(2), Integer(3), Integer(4)]:
        for point in [x, Integer(3), Rational(5, 2)]:
            ours, theirs = multigamma(point, order), sympy.multigamma(sympy_value(point), sympy_value(order))
            if not same(ours, theirs):
                different.append(('multigamma', str(ours), str(theirs)))
    with pytest.raises(ValueError):
        multigamma(x, Integer(0))
    assert not different, different[:10]
    assert stieltjes(n, Integer(-3)) == zoo
