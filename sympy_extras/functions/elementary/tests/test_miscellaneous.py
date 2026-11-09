"""The typed Min and Max against SymPy (the oracle), compared without
letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import E, Integer, Rational, Symbol, oo, pi
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.power import sqrt
from sympy_extras.functions import Max, Min, floor, max_, min_

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
p, q = Symbol('p', positive=True), Symbol('q', positive=True)
n = Symbol('n', negative=True)
k = Symbol('k', integer=True)
nn = Symbol('nn', nonnegative=True)


def atoms() -> list[Expr]:
    return [Integer(0), Integer(1), Integer(-2), Rational(1, 2), pi, Integer(3), E, sqrt(Integer(2)), oo, -oo, x, y, p,
            n, k, nn, x + 1, x - 1, 2*x, -x, p + 1, -p, floor(x), p*q]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def test_pairs_and_triples_match_sympy() -> None:
    different: list[tuple[str, str, str, str]] = []
    values = atoms()
    combinations = list(itertools.combinations(values, 2)) + list(itertools.combinations(values[::3], 3))
    for args in combinations:
        sympy_args = [sympy_value(a) for a in args]
        for ours_f, theirs_f, name in ((max_, sympy.Max, 'max'), (min_, sympy.Min, 'min')):
            ours, theirs = str(to_sympy(ours_f(*args), evaluate=False)), str(theirs_f(*sympy_args))
            if ours != theirs:
                different.append((name, str(args), ours, theirs))
    assert not different, different


def test_nested_match_sympy() -> None:
    cases: list[tuple[Expr, sympy.Expr]] = []
    sx, sy, sz = sympy.symbols('x y z')
    cases.append((min_(x, max_(y, min_(x, z))), sympy.Min(sx, sympy.Max(sy, sympy.Min(sx, sz)))))
    cases.append((min_(x, max_(y, min_(z, max_(x, 2)))), sympy.Min(sx, sympy.Max(sy, sympy.Min(sz, sympy.Max(sx, 2))))))
    cases.append((min_(max_(x, y), max_(x, z)), sympy.Min(sympy.Max(sx, sy), sympy.Max(sx, sz))))
    cases.append((max_(min_(x, y), min_(x, z), min_(x, y, z)), sympy.Max(sympy.Min(sx, sy), sympy.Min(sx, sz),
                                                                        sympy.Min(sx, sy, sz))))
    cases.append((min_(3, max_(x, 5)), sympy.Min(3, sympy.Max(sx, 5))))
    cases.append((max_(1, min_(x, 0), y), sympy.Max(1, sympy.Min(sx, 0), sy)))
    cases.append((max_(x, max_(y, x), min_(z, 2)), sympy.Max(sx, sympy.Max(sy, sx), sympy.Min(sz, 2))))
    different = [(str(ours), str(theirs)) for ours, theirs in cases if str(to_sympy(ours, evaluate=False)) != str(theirs)]
    assert not different, different


def test_facts_agree_with_sympy() -> None:
    names = ['real', 'integer', 'finite', 'negative', 'nonnegative', 'positive', 'nonpositive', 'zero', 'rational',
             'extended_real']
    for a, b in itertools.combinations(atoms(), 2):
        for node in (Max(a, b), Min(a, b)):
            s = sympy_value(node)
            for name in names:
                got = node._ask(name)
                if got is not None:
                    assert got == getattr(s, 'is_' + name), (node, name, got)


def test_nodes_and_numbers() -> None:
    node = Max(Integer(1), Integer(2))
    assert node.args == (Integer(1), Integer(2)) and node.doit() == 2 and max_(1, 2) == 2
    assert Min(x, x).args == (x, x) and min_(x, x) == x
    assert str(max_(pi, E, sqrt(Integer(2))).evalf(10)) == '3.141592654'
    assert str(Max(pi, x).subs(x, 4).evalf(5)) == '4.0000'
    assert max_(x**2 + 1, 1).diff(x) != 0
