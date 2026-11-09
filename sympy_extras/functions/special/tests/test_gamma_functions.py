"""The typed gamma functions against SymPy (the oracle), compared without
letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Float, nan
from sympy_extras.functions import (Gamma, LogGamma, PolyGamma, binomial, digamma, factorial, gamma, loggamma,
                                    polygamma, trigamma)

x = Symbol('x')
n = Symbol('n', integer=True)
m = Symbol('m', integer=True, positive=True)
q = Symbol('q', noninteger=True)
p = Symbol('p', positive=True)
k = Symbol('k', integer=True, nonpositive=True)


def values() -> list[Expr]:
    return [Integer(0), Integer(1), Integer(2), Integer(6), Integer(-1), Integer(-4), Rational(1, 2), Rational(7, 2),
            Rational(-1, 2), Rational(-5, 2), Rational(1, 3), Float('2.5'), Float('3.0'), oo, -oo, nan, I, I*oo, pi,
            x, n, m, q, p, k, x + 1]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def test_unary_match_sympy() -> None:
    different = []
    for a in values():
        for ours_f, theirs_f, name in ((gamma, sympy.gamma, 'gamma'), (loggamma, sympy.loggamma, 'loggamma'),
                                       (digamma, sympy.digamma, 'digamma'), (trigamma, sympy.trigamma, 'trigamma')):
            ours, theirs = str(to_sympy(ours_f(a), evaluate=False)), str(theirs_f(sympy_value(a)))
            if ours != theirs:
                different.append((name, str(a), ours, theirs))
    assert not different, different


def test_polygamma_matches_sympy() -> None:
    orders: list[Expr] = [Integer(-1), Integer(0), Integer(1), Integer(2), nan, n, m]
    different = []
    for order, a in itertools.product(orders, values()):
        theirs = str(sympy.polygamma(sympy_value(order), sympy_value(a)))
        ours = str(to_sympy(polygamma(order, a), evaluate=False))
        if ours != theirs:
            different.append((str(order), str(a), ours, theirs))
    assert not different, different


def test_facts_agree_with_sympy() -> None:
    for a in values():
        if a.is_integer and a.is_nonpositive:
            continue  # the poles: the functions are zoo there, SymPy's facts of the nodes contradict
        for node, s in [(Gamma(a), sympy.gamma(sympy_value(a), evaluate=False)),
                        (LogGamma(a), sympy.loggamma(sympy_value(a), evaluate=False)),
                        (PolyGamma(Integer(1), a), sympy.polygamma(1, sympy_value(a), evaluate=False)),
                        (PolyGamma(Integer(2), a), sympy.polygamma(2, sympy_value(a), evaluate=False))]:
            for name in ['real', 'positive', 'negative', 'complex']:
                got = node._ask(name)
                if got is not None:
                    assert got == getattr(s, 'is_' + name), (node, name)


def test_derivatives_and_numbers() -> None:
    sx = sympy.Symbol('x')
    for ours, theirs in [(gamma(x**2), sympy.gamma(sx**2)), (loggamma(x), sympy.loggamma(sx)),
                         (polygamma(2, x), sympy.polygamma(2, sx)), (factorial(x), sympy.factorial(sx)),
                         (binomial(x, 3), sympy.binomial(sx, 3)), (binomial(Integer(7), x), sympy.binomial(7, sx))]:
        assert str(to_sympy(ours.diff(x), evaluate=False)) == str(theirs.diff(sx)), ours
    for e, s in [(Gamma(Rational(1, 3)), sympy.gamma(sympy.Rational(1, 3))),
                 (polygamma(1, x).subs(x, Integer(3)), sympy.polygamma(1, 3)),
                 (LogGamma(2 + I), sympy.loggamma(2 + sympy.I)), (digamma(x).subs(x, Rational(1, 3)),
                                                                   sympy.digamma(sympy.Rational(1, 3)))]:
        value = complex(sympy_value(e.evalf(20)))
        assert abs(value - complex(sympy.N(s, 20))) < 1e-14, e
    assert str(gamma(Float('2.5'))) == str(sympy.gamma(sympy.Float('2.5')))
