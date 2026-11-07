"""The typed Si, Ci, Shi, Chi, li and Li against SymPy (the oracle),
compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

from typing import Callable

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Float, nan
from sympy_extras.functions import Chi, Ci, Li, Shi, Si, li

x = Symbol('x')
p = Symbol('p', positive=True)

FUNCTIONS: list[tuple[Callable[[Expr], Expr], Callable[[sympy.Expr], sympy.Expr]]] = [
    (Si, sympy.Si), (Ci, sympy.Ci), (Shi, sympy.Shi), (Chi, sympy.Chi), (li, sympy.li), (Li, sympy.Li)]


def values() -> list[Expr]:
    return [Integer(0), Integer(1), Integer(2), Integer(-1), Rational(1, 2), Float('0.5'), oo, -oo, nan, I, -I, x, -x,
            I*x, -I*x, 2*I*x, p, -p, x + 1, -x - 1]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def test_values_match_sympy() -> None:
    different = []
    for a in values():
        sa = sympy_value(a)
        for ours_f, theirs_f in FUNCTIONS:
            ours, theirs = str(to_sympy(ours_f(a), evaluate=False)), str(theirs_f(sa))
            if ours != theirs:
                different.append((ours_f.__name__, str(a), ours, theirs))
    assert not different, different


def test_derivatives_and_numbers() -> None:
    sx = sympy.Symbol('x')
    for f, g in FUNCTIONS:
        assert str(to_sympy(f(x**2).diff(x), evaluate=False)) == str(g(sx**2).diff(sx)), f
        for value in [Rational(1, 3), Integer(3), 2 + I]:
            ours = complex(sympy_value(f(x).subs(x, value).evalf(20)))
            assert abs(ours - complex(sympy.N(g(sympy_value(value)), 20))) < 1e-13, (f, value)
