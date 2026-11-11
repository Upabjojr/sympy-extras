"""The typed DiracDelta and Heaviside against SymPy (the oracle), compared
without letting SymPy re-evaluate the results."""
from __future__ import annotations

import pytest
import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Float, nan
from sympy_extras.functions import DiracDelta, Heaviside, dirac_delta, heaviside, max_, min_, sign

x = Symbol('x')
r, s = Symbol('r', real=True), Symbol('s', real=True)
p = Symbol('p', positive=True)
n = Symbol('n', negative=True)
nn = Symbol('nn', nonnegative=True)


def values() -> list[Expr]:
    return [Integer(0), Integer(2), Integer(-3), Rational(1, 2), Float('-0.5'), pi, -pi, oo, -oo, nan, x, -x, r, -r,
            2*r, -2*r, p, -p, n, nn, -nn, r - 1, 1 - r, r + s, -r - s, r*s, -r*s, x**2, sign(r)]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def test_values_match_sympy() -> None:
    different = []
    for a in values():
        sa = sympy_value(a)
        cases: list[tuple[Expr, sympy.Expr]] = [(heaviside(a), sympy.Heaviside(sa)),
                                                 (heaviside(a, 0), sympy.Heaviside(sa, 0)),
                                                 (heaviside(a, 1), sympy.Heaviside(sa, 1))]
        for k in range(4):
            cases.append((dirac_delta(a, k), sympy.DiracDelta(sa, k)))
        for ours, theirs in cases:
            if str(to_sympy(ours, evaluate=False)) != str(theirs):
                different.append((str(a), str(ours), str(theirs)))
    assert not different, different


def test_complex_arguments_are_refused() -> None:
    for a in [I, 2 + I, I*p]:
        for f in (heaviside, dirac_delta):
            with pytest.raises(ValueError):
                f(a)
    with pytest.raises(ValueError):
        dirac_delta(x, -1)


def test_derivatives_match_sympy() -> None:
    sx, sr = sympy.Symbol('x'), sympy.Symbol('r', real=True)
    cases: list[tuple[Expr, sympy.Expr, sympy.Symbol, Symbol]] = [
        (heaviside(x**2 - 1), sympy.Heaviside(sx**2 - 1), sx, x), (dirac_delta(x, 2), sympy.DiracDelta(sx, 2), sx, x),
        (heaviside(r)*r**2, sympy.Heaviside(sr)*sr**2, sr, r), (sign(r**3), sympy.sign(sr**3), sr, r),
        (sign(I*r), sympy.sign(sympy.I*sr), sr, r), (max_(x, r).diff(x), sympy.Max(sx, sr).diff(sx), sx, x),
        (min_(r, x, 2*x), sympy.Min(sr, sx, 2*sx), sx, x), (max_(r**2, 3, r), sympy.Max(sr**2, 3, sr), sr, r)]
    for ours, theirs, sv, v in cases:
        assert str(to_sympy(ours.diff(v), evaluate=False)) == str(theirs.diff(sv)), ours


def test_nodes_facts_and_round_trip() -> None:
    node = Heaviside(Integer(1))
    assert node.args == (Integer(1), Rational(1, 2)) and node.doit() == 1 and str(node) == 'Heaviside(1)'
    assert Heaviside(r, Heaviside(s)).args[1] == Rational(1, 2)
    assert DiracDelta(Integer(2)).doit() == 0
    for e in (Heaviside(r), DiracDelta(x, 1)):
        assert e.is_real is True and e.is_finite is True
        assert str(to_sympy(e)) == str(e)
    assert str(heaviside(r).subs(r, 0)) == '1/2' and heaviside(r, 1).subs(r, 0) == 1
