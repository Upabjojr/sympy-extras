"""The typed factorial, binomial, rf and ff against SymPy (the oracle),
compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import nan
from sympy_extras.functions import Binomial, Factorial, binomial, factorial, ff, rf

x = Symbol('x')
n = Symbol('n', integer=True)
m = Symbol('m', integer=True, nonnegative=True)
k = Symbol('k', integer=True, negative=True)
q = Symbol('q', noninteger=True)
p = Symbol('p', positive=True)


def values() -> list[Expr]:
    return [Integer(0), Integer(1), Integer(2), Integer(5), Integer(-1), Integer(-3), Rational(1, 2), Rational(-5, 2),
            Integer(25), oo, -oo, nan, I, pi, x, n, m, k, q, p, n + 1, m + 2, x - 1]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def test_factorial_matches_sympy() -> None:
    different = []
    for a in values():
        ours, theirs = str(to_sympy(factorial(a), evaluate=False)), str(sympy.factorial(sympy_value(a)))
        if ours != theirs:
            different.append((str(a), ours, theirs))
    assert not different, different


def test_binary_functions_match_sympy() -> None:
    different = []
    for a, b in itertools.product(values(), repeat=2):
        for ours_f, theirs_f, name in ((binomial, sympy.binomial, 'binomial'), (rf, sympy.rf, 'rf'),
                                       (ff, sympy.ff, 'ff')):
            theirs = str(theirs_f(sympy_value(a), sympy_value(b)))
            if 'gamma' in theirs:
                continue
            ours = str(to_sympy(ours_f(a, b), evaluate=False))
            if ours != theirs:
                different.append((name, str(a), str(b), ours, theirs))
    assert not different, different[:20]


def test_facts_agree_with_sympy() -> None:
    names = ['integer', 'positive', 'nonnegative', 'even', 'composite', 'real']
    for a in values():
        node = Factorial(a)
        s = sympy.factorial(sympy_value(a), evaluate=False)
        for name in names:
            got = node._ask(name)
            if got is not None:
                assert got == getattr(s, 'is_' + name), (node, name)
    for a, b in itertools.product(values(), repeat=2):
        node2 = Binomial(a, b)
        s2 = sympy.binomial(sympy_value(a), sympy_value(b), evaluate=False)
        for name in ['integer', 'nonnegative']:
            got, theirs = node2._ask(name), getattr(s2, 'is_' + name)
            # a numerical node may be decided here by evaluation where SymPy
            # leaves it open (binomial(0, 1/2) = 2/pi > 0); no contradiction
            if got is not None and theirs is not None:
                assert got == theirs, (node2, name)


def test_numbers() -> None:
    for e, s in [(Factorial(Rational(1, 2)), sympy.factorial(sympy.Rational(1, 2))),
                 (Binomial(Rational(7, 2), Rational(3, 2)), sympy.binomial(sympy.Rational(7, 2), sympy.Rational(3, 2))),
                 (rf(x, Rational(1, 3)).subs(x, Integer(2)), sympy.rf(2, sympy.Rational(1, 3)))]:
        ours = complex(sympy_value(e.evalf(20)))
        assert abs(ours - complex(sympy.N(s, 20))) < 1e-15, e
    assert factorial(30) == Integer(265252859812191058636308480000000)
