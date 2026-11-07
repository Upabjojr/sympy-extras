"""The typed fibonacci, lucas, harmonic and bernoulli against SymPy (the
oracle), compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import pytest
import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import Integer, Rational, Symbol, oo, to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.functions import bernoulli, fibonacci, harmonic, lucas

x, n = Symbol('x'), Symbol('n')
k = Symbol('k', integer=True)


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_values_match_sympy() -> None:
    different = []
    integers: list[Expr] = [Integer(i) for i in range(-6, 13)]
    for a in integers + [oo, n, k, Rational(1, 2)]:
        sa = sympy_value(a)
        for ours, theirs in [(fibonacci(a), sympy.fibonacci(sa)), (lucas(a), sympy.lucas(sa)),
                             (harmonic(a), sympy.harmonic(sa)), (bernoulli(a), sympy.bernoulli(sa))]:
            if not same(ours, theirs):
                different.append((str(ours), str(theirs)))
    for a, b in itertools.product(integers[4:] + [oo, n], [Integer(2), Integer(0), Integer(-1), Integer(-2), Integer(1),
                                                            Rational(1, 2), Integer(3), k]):
        sa, sb = sympy_value(a), sympy_value(b)
        if not same(harmonic(a, b), sympy.harmonic(sa, sb)):
            different.append((str(harmonic(a, b)), str(sympy.harmonic(sa, sb))))
    for degree in range(0, 7):
        for point in [x, x + 1, Integer(1), Integer(0), Integer(-1), Rational(1, 2)]:
            ours, theirs = bernoulli(degree, point), sympy.bernoulli(degree, sympy_value(point))
            if not same(ours, theirs):
                different.append((str(ours), str(theirs)))
        if degree >= 1:
            if not same(fibonacci(degree, x), sympy.fibonacci(degree, sympy.Symbol('x'))):
                different.append((str(fibonacci(degree, x)), str(sympy.fibonacci(degree, sympy.Symbol('x')))))
    assert not different, different
    with pytest.raises(ValueError):
        fibonacci(0, x)


def test_derivatives_and_numbers() -> None:
    sn = sympy.Symbol('n')
    assert same(harmonic(n).diff(n), sympy.harmonic(sn).diff(sn))
    assert same(harmonic(n, 3).diff(n), sympy.harmonic(sn, 3).diff(sn))
    for e, s in [(harmonic(n).subs(n, Rational(1, 2)), sympy.harmonic(sympy.Rational(1, 2))),
                 (harmonic(n, Rational(1, 2)).subs(n, Rational(5, 2)), sympy.harmonic(sympy.Rational(5, 2), sympy.Rational(1, 2)))]:
        assert abs(complex(sympy_value(e.evalf(20))) - complex(sympy.N(s, 20))) < 1e-13, e
