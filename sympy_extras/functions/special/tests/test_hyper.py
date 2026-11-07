"""The typed hyper and meijerg against SymPy (the oracle), compared
without letting SymPy re-evaluate the results."""
from __future__ import annotations

import pytest
import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import Integer, Rational, Symbol, oo, to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.interop import from_sympy
from sympy_extras.functions import Hyper, hyper, meijerg

a, b, c, z, x = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('z'), Symbol('x')


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_construction_and_radius_match_sympy() -> None:
    sa, sb, sc, sz = sympy.symbols('a b c z')
    cases: list[tuple[list[Expr], list[Expr], list[sympy.Expr], list[sympy.Expr]]] = [
        ([b, a], [c], [sb, sa], [sc]), ([a, c], [c], [sa, sc], [sc]), ([a, a, c], [a, c], [sa, sa, sc], [sa, sc]),
        ([], [], [], []), ([Integer(-2), a], [c], [sympy.Integer(-2), sa], [sc]), ([Integer(-2)], [Integer(-3)], [sympy.Integer(-2)], [sympy.Integer(-3)]),
        ([Integer(-3)], [Integer(-2)], [sympy.Integer(-3)], [sympy.Integer(-2)]), ([a, b, c], [Integer(1)], [sa, sb, sc], [sympy.Integer(1)]), ([a], [b, c], [sa], [sb, sc])]
    for ap, bq, sap, sbq in cases:
        ours, theirs = hyper(ap, bq, z), sympy.hyper(sap, sbq, sz)
        assert same(ours, theirs), (ours, theirs)
        assert isinstance(ours, Hyper)
        assert str(to_sympy(ours.radius_of_convergence)) == str(theirs.radius_of_convergence), ours
    ours_g = meijerg([b, a], [c], [Integer(1)], [], z)
    assert same(ours_g, sympy.meijerg([sb, sa], [sc], [1], [], sz))
    with pytest.raises(ValueError):
        meijerg([Integer(3)], [], [Integer(1)], [], z)
    with pytest.raises(ValueError):
        meijerg([oo], [], [], [], z)


def test_derivatives_numbers_round_trip() -> None:
    sa, sb, sc, sz, sx = sympy.symbols('a b c z x')
    for ours, theirs in [(hyper([a, b], [c], x**2), sympy.hyper([sa, sb], [sc], sx**2)),
                         (hyper([a, x], [c], x), sympy.hyper([sa, sx], [sc], sx)),
                         (meijerg([a], [b], [c], [], x**2), sympy.meijerg([sa], [sb], [sc], [], sx**2)),
                         (meijerg([], [b], [c], [], x), sympy.meijerg([], [sb], [sc], [], sx)),
                         (meijerg([], [], [], [b], x), sympy.meijerg([], [], [], [sb], sx))]:
        assert same(ours.diff(x), theirs.diff(sx)), ours
        assert from_sympy(to_sympy(ours)) == ours
    for ours, theirs in [(hyper([Rational(1, 2), 1], [Rational(3, 2)], Rational(1, 4)),
                          sympy.hyper([sympy.Rational(1, 2), 1], [sympy.Rational(3, 2)], sympy.Rational(1, 4))),
                         (meijerg([], [], [0], [], Rational(1, 3)), sympy.meijerg([], [], [0], [], sympy.Rational(1, 3)))]:
        assert abs(complex(sympy_value(ours.evalf(20))) - complex(sympy.N(theirs, 20))) < 1e-14, ours
    f = hyper([a, b], [c], z)
    assert same(f.subs(a, c), sympy.hyper([sympy.Symbol('c'), sympy.Symbol('b')], [sympy.Symbol('c')], sympy.Symbol('z')))
