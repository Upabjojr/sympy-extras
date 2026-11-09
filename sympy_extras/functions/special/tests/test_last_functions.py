"""The typed hn1, hn2, marcumq, Chebyshev roots, jacobi_normalized,
tribonacci, motzkin and andre against SymPy (the oracle), compared without
letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import pytest
import sympy
import sympy.functions as sympy_functions

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import Integer, Rational, Symbol, oo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import nan
from sympy_extras.functions import (Andre, andre, chebyshevt_root, chebyshevu_root, hn1, hn2, jacobi_normalized,
                                    marcumq, motzkin, tribonacci)

x, a, b, m = Symbol('x'), Symbol('a'), Symbol('b'), Symbol('m')
k = Symbol('k', integer=True, nonnegative=True)


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_sequences_match_sympy() -> None:
    different: list[tuple[str, ...]] = []
    for i in range(0, 15):
        for ours, theirs in [(tribonacci(i), sympy.tribonacci(i)), (motzkin(i), sympy.motzkin(i)),
                             (tribonacci(i, x), sympy.tribonacci(i, sympy.Symbol('x')))]:
            if not same(ours, theirs):
                different.append((str(ours), str(theirs)))
    for value in [Integer(i) for i in range(-7, 12)] + [oo, nan, k, x]:
        if not same(andre(value), sympy.andre(sympy_value(value))):
            different.append(('andre', str(value), str(andre(value)), str(sympy.andre(sympy_value(value)))))
        node, s = Andre(value), sympy.andre(sympy_value(value), evaluate=False)
        for name in ['integer', 'positive']:
            got = node._ask(name)
            if got is not None:
                assert got == getattr(s, 'is_' + name), (node, name)
    assert not different, different
    for bad in [Integer(-1)]:
        with pytest.raises(ValueError):
            tribonacci(bad)
        with pytest.raises(ValueError):
            motzkin(bad)


def test_bessel_leftovers_match_sympy() -> None:
    sx, sa, sb, sm = sympy.symbols('x a b m')
    for f, g in [(hn1, sympy_functions.hn1), (hn2, sympy_functions.hn2)]:
        assert same(f(m, x), g(sm, sx)) and same(f(m, x**2).diff(x), g(sm, sx**2).diff(sx))
        value = complex(sympy_value(f(Integer(1), Rational(1, 3)).evalf(20)))
        assert abs(value - complex(sympy.N(g(1, sympy.Rational(1, 3)), 20))) < 1e-13, f
    different = []
    for first, second, third in itertools.product([Integer(0), Integer(1), Integer(2), m], [Integer(0), a, b],
                                                  [Integer(0), a, b]):
        ours = marcumq(first, second, third)
        theirs = sympy.marcumq(sympy_value(first), sympy_value(second), sympy_value(third))
        if not same(ours, theirs):
            different.append((str(ours), str(theirs)))
    assert not different, different
    for v, sv in [(a, sa), (b, sb)]:
        assert same(marcumq(m, a, b).diff(v), sympy.marcumq(sm, sa, sb).diff(sv)), v


def test_roots_and_normalized_jacobi_match_sympy() -> None:
    for n_value, k_value in [(3, 0), (3, 1), (3, 2), (5, 3), (4, 0)]:
        assert same(chebyshevt_root(n_value, k_value), sympy.chebyshevt_root(n_value, k_value))
        assert same(chebyshevu_root(n_value, k_value), sympy.chebyshevu_root(n_value, k_value))
    with pytest.raises(ValueError):
        chebyshevt_root(3, 3)
    sx, sa, sb = sympy.symbols('x a b')
    for degree in [Integer(0), Integer(1), Integer(2)]:
        assert same(jacobi_normalized(degree, a, b, x), sympy.jacobi_normalized(sympy_value(degree), sa, sb, sx)), degree
    assert same(jacobi_normalized(Integer(1), x, x, x), sympy.jacobi_normalized(1, sx, sx, sx))
