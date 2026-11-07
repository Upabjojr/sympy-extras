"""The typed sinc, Rem, subfactorial and factorial2 against SymPy (the
oracle), compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import pytest
import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, to_sympy, zoo
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Float, nan
from sympy_extras.functions import Factorial2, Sinc, Subfactorial, factorial2, rem, sinc, subfactorial

x = Symbol('x')
r = Symbol('r', real=True)
n = Symbol('n', integer=True)
m = Symbol('m', integer=True, nonnegative=True)


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_sinc_matches_sympy() -> None:
    different = []
    values: list[Expr] = [Integer(0), oo, -oo, nan, zoo, x, -x, r, I*r, pi, 2*pi, -3*pi, pi/2, 3*pi/2, -5*pi/2, pi/3,
                          n*pi, Integer(2), Float('0.5')]
    for a in values:
        if not same(sinc(a), sympy.sinc(sympy_value(a))):
            different.append((str(sinc(a)), str(sympy.sinc(sympy_value(a)))))
        node = Sinc(a)
        s = sympy.sinc(sympy_value(a), evaluate=False)
        for name in ['real', 'finite', 'zero']:
            got = node._ask(name)
            if got is not None:
                assert got == getattr(s, 'is_' + name), (node, name)
    assert not different, different
    assert same(sinc(x**2).diff(x), sympy.sinc(sympy.Symbol('x')**2).diff(sympy.Symbol('x')))


def test_rem_matches_sympy() -> None:
    different = []
    values: list[Expr] = [Integer(7), Integer(-7), Integer(3), Integer(-3), Rational(7, 2), Float('2.5'), Integer(0),
                          x, -x, n, oo, nan, Integer(1)]
    for p, q in itertools.product(values, repeat=2):
        sp, sq = sympy_value(p), sympy_value(q)
        try:
            theirs = sympy.Rem(sp, sq)
        except ZeroDivisionError:
            with pytest.raises(ZeroDivisionError):
                rem(p, q)
            continue
        if not same(rem(p, q), theirs):
            different.append((str(p), str(q), str(rem(p, q)), str(theirs)))
    assert not different, different


def test_subfactorial_factorial2_match_sympy() -> None:
    different = []
    for a in [Integer(i) for i in range(-7, 12)] + [oo, nan, x, n, m, 2*m, 2*m + 1, Rational(1, 2)]:
        sa = sympy_value(a)
        for ours_f, theirs_f, cls in [(subfactorial, sympy.subfactorial, Subfactorial),
                                      (factorial2, sympy.factorial2, Factorial2)]:
            try:
                theirs = theirs_f(sa)
            except ValueError:
                with pytest.raises(ValueError):
                    ours_f(a)
                continue
            if not same(ours_f(a), theirs):
                different.append((ours_f.__name__, str(a), str(ours_f(a)), str(theirs)))
            if not a.is_Number:
                node = cls(a)
                s = theirs_f(sa, evaluate=False)
                for name in ['integer', 'even', 'odd', 'positive', 'nonnegative']:
                    got = node._ask(name)
                    if got is not None:
                        assert got == getattr(s, 'is_' + name), (node, name)
    assert not different, different
