"""The typed floor, ceiling and frac against SymPy (the oracle), compared
without letting SymPy re-evaluate the results."""
from __future__ import annotations

from typing import Callable

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import E, I, Integer, Rational, Symbol, oo, pi, to_sympy, zoo
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Float, nan
from sympy_extras.core.power import sqrt
from sympy_extras.functions import Ceiling, Floor, Frac, ceiling, exp, floor, frac, log, sin

x, y = Symbol('x'), Symbol('y')
r = Symbol('r', real=True)
n = Symbol('n', integer=True)
p = Symbol('p', positive=True)

FUNCTIONS: list[tuple[Callable[[Expr], Expr], Callable[[sympy.Expr], sympy.Expr]]] = [
    (floor, sympy.floor), (ceiling, sympy.ceiling), (frac, sympy.frac)]


def arguments() -> list[Expr]:
    values: list[Expr] = [Integer(0), Integer(3), Integer(-2), Rational(7, 2), Rational(-7, 2), Rational(1, 3),
                          Float('2.5'), Float('-0.25'), pi, -pi, E, -E, sqrt(Integer(2)), -sqrt(Integer(3)),
                          pi/4, -pi/4, 2*pi, I, 2*I, Rational(5, 2)*I, -pi*I, 3 + 2*I, Rational(1, 2) + I*pi,
                          nan, zoo, oo, -oo, x, -x, r, n, 2*n, n + Rational(5, 2), x + 3, x - Rational(1, 2),
                          x + pi, r + I, I*r, I*x, x + y, floor(x), -floor(x), ceiling(x) + 1, p, -p, p/(p + 1),
                          1/(p + 1), -1/(p + 1), exp(Integer(2)), log(Integer(10)), sin(Integer(1)),
                          n + pi + I*Rational(3, 2), sqrt(Integer(2)) + I*x]
    return values


def test_values_match_sympy() -> None:
    different: list[tuple[str, str, str, str]] = []
    for a in arguments():
        sa = sympy_expr(to_sympy(a))
        for ours_f, theirs_f in FUNCTIONS:
            ours, theirs = str(to_sympy(ours_f(a), evaluate=False)), str(theirs_f(sa))
            if ours != theirs:
                different.append((ours_f.__name__, str(a), ours, theirs))
    # frac(oo), frac(-oo): SymPy answers AccumBounds(0, 1), not ported
    accepted = {('frac', 'oo'), ('frac', '-oo')}
    unexpected = [d for d in different if (d[0], d[1]) not in accepted]
    assert not unexpected, unexpected


def test_facts_agree_with_sympy() -> None:
    names = ['real', 'integer', 'finite', 'negative', 'nonnegative', 'positive', 'nonpositive', 'zero', 'imaginary']
    for a in arguments():
        if a in (oo, -oo):
            continue  # frac(oo) is AccumBounds(0, 1) in SymPy
        for f, _ in FUNCTIONS:
            node = f(a)
            s = sympy_expr(to_sympy(node))
            for name in names:
                got = node._ask(name)
                if got is not None:
                    assert got == getattr(s, 'is_' + name), (node, name, got)


def test_numbers_and_nodes() -> None:
    for f, g in FUNCTIONS:
        for value in [Rational(7, 3), -pi, 2 + Rational(5, 2)*I]:
            ours = complex(sympy_expr(to_sympy(f(x).subs(x, value).evalf(20))))
            theirs = complex(sympy.N(g(sympy_expr(to_sympy(value))), 20))
            assert abs(ours - theirs) < 1e-15, (f, value)
    for cls, function in [(Floor, floor), (Ceiling, ceiling), (Frac, frac)]:
        node = cls(Rational(7, 2))
        assert type(node) is cls and function(Rational(7, 2)) != node and node.doit() == function(Rational(7, 2))
