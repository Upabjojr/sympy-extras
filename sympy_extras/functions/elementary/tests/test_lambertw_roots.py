"""The typed LambertW, root, cbrt and real_root against SymPy (the
oracle), compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import E, I, Integer, Rational, Symbol, oo, pi
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Float, nan
from sympy_extras.functions import LambertW, cbrt, exp, lambertw, log, real_root, root

x = Symbol('x')
n = Symbol('n', integer=True)
p = Symbol('p', positive=True)
r = Symbol('r', real=True)


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def test_lambertw_matches_sympy() -> None:
    points: list[Expr] = [Integer(0), Integer(1), E, -1/E, -log(Integer(2))/2, 2*log(Integer(2)), -pi/2,
                          exp(1 + E), oo, -2*exp(Integer(-2)), x, p, nan, Float('0.5'), I]
    branches: list[Expr] = [Integer(0), Integer(-1), Integer(1), Integer(2), n]
    different = []
    for a in points:
        sa = sympy_value(a)
        cases = [(lambertw(a), sympy.LambertW(sa))] + [(lambertw(a, k), sympy.LambertW(sa, sympy_value(k)))
                                                       for k in branches]
        for ours, theirs in cases:
            if str(to_sympy(ours, evaluate=False)) != str(theirs):
                different.append((str(ours), str(theirs)))
    assert not different, different


def test_lambertw_facts_derivatives_numbers() -> None:
    names = ['extended_real', 'finite', 'zero', 'algebraic']
    for a in [Integer(2), -1/E, Rational(-1, 10), Integer(-1), p, r, x, Integer(3) + I]:
        for k in [Integer(0), Integer(-1), Integer(1), n]:
            node = LambertW(a, k)
            s = sympy.LambertW(sympy_value(a), sympy_value(k), evaluate=False)
            for name in names:
                got = node._ask(name)
                if got is not None:
                    assert got == getattr(s, 'is_' + name), (node, name)
    sx = sympy.Symbol('x')
    assert str(to_sympy(lambertw(x**2).diff(x), evaluate=False)) == str(sympy.LambertW(sx**2).diff(sx))
    assert str(to_sympy(lambertw(x, 2).diff(x), evaluate=False)) == str(sympy.LambertW(sx, 2).diff(sx))
    for a, branch in [(Integer(3), 0), (Rational(-1, 5), -1), (2 + I, 1)]:
        ours = complex(sympy_value(LambertW(a, Integer(branch)).evalf(20)))
        assert abs(ours - complex(sympy.N(sympy.LambertW(sympy_value(a), branch), 20))) < 1e-14


def test_roots_match_sympy() -> None:
    sx, sn = sympy.Symbol('x'), sympy.Symbol('n', integer=True)
    cases: list[tuple[Expr, sympy.Expr]] = [
        (cbrt(8), sympy.cbrt(8)), (cbrt(x), sympy.cbrt(sx)), (cbrt(-27), sympy.cbrt(-27)), (root(16, 4), sympy.root(16, 4)),
        (root(x, 3, 1), sympy.root(sx, 3, 1)), (root(-27, 3, 2), sympy.root(-27, 3, 2)), (root(x, n), sympy.root(sx, sn)),
        (real_root(-8, 3), sympy.real_root(-8, 3)), (real_root(-32, 5), sympy.real_root(-32, 5)),
        (real_root(x, 3), sympy.real_root(sx, 3)), (real_root(x, n), sympy.real_root(sx, sn)),
        (real_root(root(-8, 3)), sympy.real_root(sympy.root(-8, 3))),
        (real_root(x + root(-2, 3)), sympy.real_root(sx + sympy.root(-2, 3))), (real_root(16, 4), sympy.real_root(16, 4))]
    different = [(str(o), str(t)) for o, t in cases if str(to_sympy(o, evaluate=False)) != str(t)]
    assert not different, different
