"""The typed Piecewise against SymPy (the oracle), compared without
letting SymPy re-evaluate the results."""
from __future__ import annotations

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, to_sympy
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, and_, false, not_, or_, true
from sympy_extras.core.relational import eq, ge, gt, le, lt, ne
from sympy_extras.functions import Piecewise, conjugate, exp, piecewise, sin

x, y = Symbol('x'), Symbol('y')
p = Symbol('p', positive=True)
r = Symbol('r', real=True)

PairList = list[tuple[Expr, Boolean]]


def cases() -> list[PairList]:
    return [
        [(Integer(1), lt(-x, -1)), (Integer(4), lt(-x, -1)), (Integer(3), false), (Integer(2), true),
         (Integer(5), lt(x, 1))],
        [(x**2, lt(x, 0)), (x, true)],
        [(x, true)],
        [(Integer(1), lt(x, 0)), (Integer(1), gt(x, 2)), (Integer(0), true)],
        [(Integer(1), lt(-2*x + 1, 0)), (Integer(0), true)],
        [(x, lt(3*x, 6)), (y, ge(-x, -5)), (Integer(0), true)],
        [(x, lt(x, 0)), (y, lt(x, 0)), (Integer(1), true)],
        [(x, lt(x, 0)), (y, and_(ge(x, 0), lt(x, 1))), (Integer(1), true)],
        [(x, le(x, 0)), (y, and_(gt(x, 0), lt(y, 1)))],
        [(x, lt(x, 1)), (y, and_(lt(x, 1), lt(y, 2))), (Integer(2), true)],
        [(Integer(1), false)],
        [(Integer(1), gt(p, 0)), (Integer(2), true)],
        [(sin(x), eq(x, 0)), (exp(x), ne(x, 0))],
        [(Integer(1), or_(lt(x, 0), and_(gt(x, 1), lt(y, 0)))), (Integer(1), gt(y, 3)), (Integer(0), true)],
        [(Integer(3), lt(Rational(1, 2)*x + 1, 0)), (x, not_(lt(x, -2)))],
        [(Integer(1), lt(x, 0)), (Integer(2), ge(x, 0))],
        [(Integer(1), lt(x, 0)), (Integer(2), and_(ge(x, 0), le(x, 0))), (Integer(3), true)],
    ]


def test_construction_matches_sympy() -> None:
    different: list[tuple[str, str, str]] = []
    for pairs in cases():
        ours = str(to_sympy(piecewise(*pairs), evaluate=False))
        theirs = str(sympy.Piecewise(*[(to_sympy(e), to_sympy(c)) for e, c in pairs]))
        if ours != theirs:
            different.append((str(pairs), ours, theirs))
    assert not different, different


def test_nested_and_substitution_match_sympy() -> None:
    inner = piecewise((x, lt(x, 0)), (Integer(2), true))
    nested = piecewise((inner, lt(x, 0)), (Integer(3), true))
    s_inner = sympy.Piecewise((sympy.Symbol('x'), sympy.Symbol('x') < 0), (2, True))
    s_nested = sympy.Piecewise((s_inner, sympy.Symbol('x') < 0), (3, True))
    assert str(to_sympy(nested, evaluate=False)) == str(s_nested)
    f = piecewise((x**2, lt(x, 0)), (x + 1, lt(x, 3)), (Integer(7), true))
    for value in [Integer(-2), Integer(0), Integer(2), Integer(3), Integer(9)]:
        ours = f.subs(x, value)
        theirs = sympy_expr(to_sympy(f)).subs(sympy.Symbol('x'), to_sympy(value))
        assert str(to_sympy(ours, evaluate=False)) == str(theirs), value
    g = f.subs(x, y + 1)
    assert str(to_sympy(g, evaluate=False)) == str(sympy_expr(to_sympy(f)).subs(sympy.Symbol('x'),
                                                                                  sympy.Symbol('y') + 1))


def test_derivative_conjugate_and_facts() -> None:
    f = piecewise((x**2, lt(r, 0)), (sin(x), true))
    sf = sympy_expr(to_sympy(f))
    assert str(to_sympy(f.diff(x), evaluate=False)) == str(sf.diff(sympy.Symbol('x')))
    g = piecewise((x + I, lt(r, 0)), (r, true))
    assert str(to_sympy(conjugate(g), evaluate=False)) == str(sympy.conjugate(to_sympy(g)))
    h = piecewise((p, lt(r, 0)), (p + 1, true))
    sh = sympy_expr(to_sympy(h))
    for name in ['positive', 'real', 'finite', 'integer', 'zero', 'negative']:
        got = h._ask(name)
        if got is not None:
            assert got == getattr(sh, 'is_' + name), name
    assert h.is_positive is True


def test_round_trip_and_node() -> None:
    node = Piecewise((x, true))
    assert node.args[0].expr == x and node.doit() == x and piecewise((x, true)) == x
    f = piecewise((x**2, lt(x, 0)), (x, true))
    back = Basic.__eq__(f, f)
    assert back is True
    from sympy_extras.core.interop import from_sympy
    assert from_sympy(to_sympy(f)) == f
