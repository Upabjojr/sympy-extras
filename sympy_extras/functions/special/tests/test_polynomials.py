"""The typed orthogonal polynomials against SymPy (the oracle), compared
without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools
from typing import Callable

import pytest
import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.functions import assoc_laguerre, chebyshevt, chebyshevu, conjugate, hermite, laguerre, legendre

x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
n = Symbol('n', integer=True)
m = Symbol('m', integer=True, positive=True)
s = Symbol('s')

FUNCTIONS: list[tuple[Callable[[Expr, Expr], Expr], Callable[[sympy.Expr, sympy.Expr], sympy.Expr]]] = [
    (legendre, sympy.legendre), (chebyshevt, sympy.chebyshevt), (chebyshevu, sympy.chebyshevu),
    (hermite, sympy.hermite), (laguerre, sympy.laguerre)]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_values_match_sympy() -> None:
    orders: list[Expr] = [Integer(k) for k in range(-4, 8)] + [n, -n, m, -m, -n - 1, -m - 2, s]
    points: list[Expr] = [x, -x, x + 1, 2*y, Integer(0), Integer(1), Integer(-1), Rational(1, 2), oo, -oo, I]
    different = []
    for order, z in itertools.product(orders, points):
        so, sz = sympy_value(order), sympy_value(z)
        for f, g in FUNCTIONS:
            try:
                theirs = g(so, sz)
            except ValueError:
                with pytest.raises(ValueError):
                    f(order, z)
                continue
            if not same(f(order, z), theirs):
                different.append((f.__name__, str(order), str(z), str(f(order, z)), str(theirs)))
    for k in range(5):
        for alpha in [a, Integer(0), Integer(2), Rational(1, 2)]:
            for z in [x, Integer(0), x + 1]:
                ours, theirs = assoc_laguerre(k, alpha, z), sympy.assoc_laguerre(k, sympy_value(alpha), sympy_value(z))
                if not same(ours, theirs):
                    different.append(('assoc_laguerre', str(k), str((alpha, z)), str(ours), str(theirs)))
    for z in [Integer(0), oo, -oo, x]:
        ours, theirs = assoc_laguerre(m, a, z), sympy.assoc_laguerre(sympy_value(m), sympy_value(a), sympy_value(z))
        if not same(ours, theirs):
            different.append(('assoc_laguerre', 'm', str(z), str(ours), str(theirs)))
    assert not different, different[:20]


def test_derivatives_conjugates_numbers() -> None:
    sx, sn = sympy.Symbol('x'), sympy.Symbol('n', integer=True)
    for f, g in FUNCTIONS:
        assert same(f(n, x**2).diff(x), g(sn, sx**2).diff(sx)), f
        assert same(conjugate(f(n, x)), sympy.conjugate(g(sn, sx))), f
        ours = complex(sympy_value(f(n, x).subs(n, Integer(5)).subs(x, Rational(1, 3)).evalf(20)))
        assert abs(ours - complex(sympy.N(g(sympy.Integer(5), sympy.Rational(1, 3)), 20))) < 1e-14, f
    sa = sympy.Symbol('a')
    assert same(assoc_laguerre(n, a, x).diff(x), sympy.assoc_laguerre(sn, sa, sx).diff(sx))


def test_jacobi_gegenbauer_assoc_legendre_hermite_prob_match_sympy() -> None:
    from sympy_extras.functions import assoc_legendre, gegenbauer, hermite_prob, jacobi
    b = Symbol('b')
    p = Symbol('p', positive=True)
    parameters: list[Expr] = [Integer(0), Integer(1), Integer(-1), Rational(1, 2), Rational(-1, 2), a, -a, b, Integer(3)]
    points: list[Expr] = [x, -x, Integer(0), Integer(1), Integer(-1), oo, x + 1]
    orders: list[Expr] = [Integer(k) for k in range(0, 4)] + [n, m, -m, s]
    different = []

    def check(name: str, ours: Callable[[], Expr], theirs: Callable[[], sympy.Basic]) -> None:
        try:
            expected = theirs()
        except ValueError:
            with pytest.raises(ValueError):
                ours()
            return
        value = ours()
        if not same(value, expected):
            different.append((name, str(value), str(expected)))

    for order, alpha, z in itertools.product(orders, parameters, points):
        so, sa, sz = sympy_value(order), sympy_value(alpha), sympy_value(z)
        check('gegenbauer', lambda: gegenbauer(order, alpha, z), lambda: sympy.gegenbauer(so, sa, sz))
        for beta in [alpha, -alpha, b, Integer(2)]:
            sbeta = sympy_value(beta)
            check('jacobi', lambda: jacobi(order, alpha, beta, z), lambda: sympy.jacobi(so, sa, sbeta, sz))
    for degree, order2, z in itertools.product(orders, [Integer(k) for k in range(-3, 4)] + [m, -m, p], points):
        sd, so2, sz = sympy_value(degree), sympy_value(order2), sympy_value(z)
        check('assoc_legendre', lambda: assoc_legendre(degree, order2, z), lambda: sympy.assoc_legendre(sd, so2, sz))
    for order, z in itertools.product(orders + [Integer(-2)], points):
        so, sz = sympy_value(order), sympy_value(z)
        check('hermite_prob', lambda: hermite_prob(order, z), lambda: sympy.hermite_prob(so, sz))
    assert not different, different[:20]
    sx, sn, sa, sb = sympy.Symbol('x'), sympy.Symbol('n', integer=True), sympy.Symbol('a'), sympy.Symbol('b')
    for ours, theirs in [(jacobi(n, a, b, x), sympy.jacobi(sn, sa, sb, sx)), (gegenbauer(n, a, x), sympy.gegenbauer(sn, sa, sx)),
                         (assoc_legendre(n, m, x), sympy.assoc_legendre(sn, sympy_value(m), sx)),
                         (hermite_prob(n, x), sympy.hermite_prob(sn, sx))]:
        assert same(ours.diff(x), theirs.diff(sx)), ours
        assert same(conjugate(ours), sympy.conjugate(theirs)), ours
