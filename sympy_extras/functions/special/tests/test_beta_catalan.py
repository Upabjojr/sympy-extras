"""The typed beta and catalan against SymPy (the oracle), compared
without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Float, nan
from sympy_extras.functions import Beta, Catalan, beta, catalan

x, y = Symbol('x'), Symbol('y')
n = Symbol('n', integer=True)
m = Symbol('m', integer=True, nonnegative=True)
p = Symbol('p', positive=True)
q = Symbol('q', noninteger=True, negative=True)


def values() -> list[Expr]:
    return [Integer(0), Integer(1), Integer(2), Integer(3), Integer(-1), Integer(-2), Integer(-5), Rational(1, 2),
            Rational(3, 2), Rational(-1, 2), Rational(-3, 2), Rational(5, 2), Float('2.5'), oo, nan, I, x, n, m, p, q]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def test_catalan_matches_sympy() -> None:
    different = []
    for a in values():
        ours, theirs = str(to_sympy(catalan(a), evaluate=False)), str(sympy.catalan(sympy_value(a)))
        if ours != theirs:
            different.append((str(a), ours, theirs))
    assert not different, different


def test_beta_matches_sympy() -> None:
    different = []
    for a in values():
        if str(to_sympy(beta(a), evaluate=False)) != str(sympy.beta(sympy_value(a))):
            different.append((str(a), str(beta(a)), str(sympy.beta(sympy_value(a)))))
    for a, b in itertools.product(values(), repeat=2):
        ours, theirs = str(to_sympy(beta(a, b), evaluate=False)), str(sympy.beta(sympy_value(a), sympy_value(b)))
        if ours != theirs:
            different.append((str((a, b)), ours, theirs))
    assert not different, different[:20]


def test_facts_derivatives_numbers() -> None:
    for a in values():
        node = Catalan(a)
        s = sympy.catalan(sympy_value(a), evaluate=False)
        for name in ['integer', 'positive', 'composite']:
            got = node._ask(name)
            if got is not None:
                assert got == getattr(s, 'is_' + name), (node, name)
    for a, b in itertools.product([p, x, I, Integer(2)], repeat=2):
        got = Beta(a, b)._ask('real')
        if got is not None:
            assert got == sympy.beta(sympy_value(a), sympy_value(b), evaluate=False).is_real, (a, b)
    sx, sy = sympy.Symbol('x'), sympy.Symbol('y')
    for v, sv in [(x, sx), (y, sy)]:
        assert str(to_sympy(beta(x, y).diff(v), evaluate=False)) == str(sympy.beta(sx, sy).diff(sv))
    assert str(to_sympy(catalan(x**2).diff(x), evaluate=False)) == str(sympy.catalan(sx**2).diff(sx))
    for e, s in [(Beta(Rational(1, 3), Rational(2, 7)), sympy.beta(sympy.Rational(1, 3), sympy.Rational(2, 7))),
                 (Catalan(Rational(1, 2)), sympy.catalan(sympy.Rational(1, 2)))]:
        assert abs(complex(sympy_value(e.evalf(20))) - complex(sympy.N(s, 20))) < 1e-14, e
