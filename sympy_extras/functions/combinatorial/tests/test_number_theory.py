"""The typed number-theoretic functions and the aliases against SymPy (the
oracle), compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools
from typing import Callable

import pytest
import sympy
import sympy.functions as sympy_functions

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import Integer, Rational, Symbol, oo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Float
from sympy_extras.functions import (E1, Eijk, Mobius, divisor_sigma, identity_function, jacobi_symbol, kronecker_symbol,
                                    legendre_symbol, ln, mobius, primenu, primeomega, primepi, reduced_totient, totient,
                                    udivisor_sigma)

n = Symbol('n', integer=True, positive=True)
p = Symbol('p', prime=True)
k = Symbol('k', integer=True, nonnegative=True)
x, z = Symbol('x'), Symbol('z')

UNARY: list[tuple[Callable[[Expr], Expr], Callable[[sympy.Expr], sympy.Expr]]] = [
    (totient, sympy.totient), (reduced_totient, sympy.reduced_totient), (primenu, sympy.primenu),
    (primeomega, sympy.primeomega), (mobius, sympy.mobius), (divisor_sigma, sympy.divisor_sigma),
    (udivisor_sigma, sympy_functions.udivisor_sigma)]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def compare(name: str, ours: Callable[[], Expr], theirs: Callable[[], sympy.Basic], different: list[tuple[str, ...]]) -> None:
    try:
        expected = theirs()
    except (ValueError, TypeError) as error:
        with pytest.raises(type(error)):
            ours()
        return
    value = ours()
    if not same(value, expected):
        different.append((name, str(value), str(expected)))


def test_unary_functions_match_sympy() -> None:
    different: list[tuple[str, ...]] = []
    values: list[Expr] = [Integer(i) for i in range(-2, 40)] + [Integer(1024), Integer(720720), n, p, Rational(1, 2), x,
                                                               p**2, n**2, 6*p, Integer(30)*n]
    # sympy/sympy#31275 (V75) and its neighbours: SymPy's mobius of a product with symbols is wrong
    # at p = 1 or n = 1 (mobius(n**2) = 0) and forgets the symbolic factors (mobius(6*p) = 1)
    symbolic_products = [str(v) for v in values[-3:]]
    for a in values:
        sa = sympy_value(a)
        for ours_f, theirs_f in UNARY:
            if ours_f is mobius and str(a) in symbolic_products:
                assert ours_f(a) == Mobius(a), a
                continue
            compare(ours_f.__name__, lambda: ours_f(a), lambda: theirs_f(sa), different)
    for a in values[:20] + [oo, -oo, Float('10.5'), Rational(31, 2), x]:
        sa = sympy_value(a)
        compare('primepi', lambda: primepi(a), lambda: sympy.primepi(sa), different)
    assert not different, different[:20]


def test_mobius_of_products_with_symbols() -> None:
    # sympy/sympy#31275 (V75): mu(1) = 1, so n**2 is not a square factor for a positive integer n
    # which may be 1; and a symbolic factor may share a prime with the others (mobius(3*n) at n = 3)
    q = Symbol('q', integer=True, positive=True)
    j = Symbol('j', integer=True, positive=True)
    for e in (q**2, q**(j + 1), 3*q, 6*q, 3*q**2, 2*Integer(3)**j):
        assert mobius(e) == Mobius(e), e
    assert mobius(q**2).subs(q, Integer(1)) == 1 and mobius(3*q).subs(q, Integer(3)) == 0
    assert mobius(4*q) == 0 and mobius(Integer(3)**(j + 1)) == 0 and mobius(12*q**2) == 0
    assert mobius((q + 1)**2) == 0 and mobius((q + 1)**(j + 1)) == 0 and mobius(p**2) == 0
    assert [mobius(Integer(k)) for k in (1, 2, 4, 6, 30, 12)] == [1, -1, 0, 1, -1, 0]


def test_sigma_and_symbols_match_sympy() -> None:
    different: list[tuple[str, ...]] = []
    firsts: list[Expr] = [Integer(1), Integer(12), Integer(36), Integer(7), p, n]
    seconds: list[Expr] = [Integer(0), Integer(2), k, Integer(-1), Rational(1, 2)]
    for a, b in itertools.product(firsts, seconds):
        sa, sb = sympy_value(a), sympy_value(b)
        compare('divisor_sigma', lambda: divisor_sigma(a, b), lambda: sympy.divisor_sigma(sa, sb), different)
        compare('udivisor_sigma', lambda: udivisor_sigma(a, b), lambda: sympy_functions.udivisor_sigma(sa, sb), different)
    tops: list[Expr] = [Integer(i) for i in range(-6, 12)] + [n]
    bottoms: list[Expr] = [Integer(i) for i in (-8, -3, -1, 0, 1, 2, 3, 5, 7, 8, 9, 15, 21)] + [p]
    for a, b in itertools.product(tops, bottoms):
        sa, sb = sympy_value(a), sympy_value(b)
        compare('legendre', lambda: legendre_symbol(a, b), lambda: sympy.legendre_symbol(sa, sb), different)
        compare('jacobi', lambda: jacobi_symbol(a, b), lambda: sympy.jacobi_symbol(sa, sb), different)
        compare('kronecker', lambda: kronecker_symbol(a, b), lambda: sympy.kronecker_symbol(sa, sb), different)
    assert not different, different[:20]
    for f, g in UNARY:
        node = f(n)
        s = g(sympy.Symbol('n', integer=True, positive=True))
        for name in ['integer', 'positive', 'nonnegative', 'real']:
            got = node._ask(name)
            if got is not None:
                assert got == getattr(s, 'is_' + name), (node, name)


def test_aliases() -> None:
    assert ln(x) == ln(x) and str(ln(x)) == 'log(x)'
    assert same(E1(z), sympy.E1(sympy.Symbol('z')))
    assert Eijk(1, 2, 3) == 1 and Eijk(2, 1, 3) == -1
    assert str(identity_function()) == str(sympy.Id)
