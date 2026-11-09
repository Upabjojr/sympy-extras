"""The typed B-splines, appellf1 and jn_zeros against SymPy (the oracle),
compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import sympy
from sympy.functions.special.bessel import jn_zeros as sympy_jn_zeros

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.functions import appellf1, bspline_basis, bspline_basis_set, jn_zeros

x = Symbol('x')
a, b1, b2, c, y = Symbol('a'), Symbol('b1'), Symbol('b2'), Symbol('c'), Symbol('y')


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_bsplines_match_sympy() -> None:
    sx = sympy.Symbol('x')
    different = []
    knot_sets: list[tuple[int, ...]] = [(0, 1, 2, 3, 4), (0, 0, 1, 2, 3, 3), (0, 0, 0, 1, 1, 1), (0, 2, 3, 7, 8, 9)]
    for knots in knot_sets:
        for d in range(0, 4):
            if len(knots) - d - 1 < 1:
                continue
            ours = bspline_basis_set(d, knots, x)
            theirs = sympy.bspline_basis_set(d, knots, sx)
            for o, t in zip(ours, theirs):
                if not same(o, t):
                    different.append((str(knots), d, str(o), str(t)))
    assert not different, different[:5]
    assert same(bspline_basis(2, (0, Rational(1, 2), 1, 2), 0, x), sympy.bspline_basis(2, (0, sympy.Rational(1, 2), 1, 2), 0, sx))


def test_appellf1_and_jn_zeros_match_sympy() -> None:
    sa, sb1, sb2, sc, sx, sy = sympy.symbols('a b1 b2 c x y')
    for args, sargs in [((a, b2, b1, c, x, y), (sa, sb2, sb1, sc, sx, sy)), ((a, b1, b1, c, y, x), (sa, sb1, sb1, sc, sy, sx)),
                        ((a, b1, b2, c, Integer(0), Integer(0)), (sa, sb1, sb2, sc, 0, 0))]:
        assert same(appellf1(*args), sympy.appellf1(*sargs)), args
    e, t = appellf1(a, b1, b2, c, x, y), sympy.appellf1(sa, sb1, sb2, sc, sx, sy)
    for v, sv in [(x, sx), (y, sy), (a, sa)]:
        assert same(e.diff(v), t.diff(sv)), v
    value = complex(sympy_value(appellf1(Integer(1), Integer(2), Integer(3), Integer(5), Rational(1, 2), Rational(1, 4)).evalf(15)))
    assert abs(value - complex(sympy.N(sympy.appellf1(1, 2, 3, 5, sympy.Rational(1, 2), sympy.Rational(1, 4)), 15))) < 1e-12
    for n in range(0, 4):
        assert [str(z) for z in jn_zeros(n, 4)] == [str(z) for z in sympy_jn_zeros(n, 4)], n
