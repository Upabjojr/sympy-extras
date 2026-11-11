"""The typed spherical harmonics, KroneckerDelta and LeviCivita against
SymPy (the oracle), compared without letting SymPy re-evaluate the
results."""
from __future__ import annotations

import itertools

import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import Integer, Rational, Symbol, pi
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.functions import KroneckerDelta, LeviCivita, Ynm, Ynm_c, Znm, conjugate

n, m, t, p = Symbol('n'), Symbol('m'), Symbol('theta'), Symbol('phi')
mp = Symbol('mp', positive=True)
mn = Symbol('mn', negative=True)
i, j, k = Symbol('i', integer=True), Symbol('j', integer=True), Symbol('k')


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_harmonics_match_sympy() -> None:
    different = []
    orders: list[Expr] = [m, -m, mp, -mp, mn, Integer(0), Integer(2), Integer(-1)]
    angles: list[Expr] = [t, -t, 2*t, Integer(0), pi/3, -pi/4]
    for order, theta, phi in itertools.product(orders, angles, [p, -p, Integer(0)]):
        so, st, sp = sympy_value(order), sympy_value(theta), sympy_value(phi)
        sn = sympy_value(n)
        for ours, theirs in [(Ynm(n, order, theta, phi), sympy.Ynm(sn, so, st, sp)),
                             (Znm(n, order, theta, phi), sympy.Znm(sn, so, st, sp)),
                             (Ynm_c(n, order, theta, phi), sympy.Ynm_c(sn, so, st, sp))]:
            if not same(ours, theirs):
                different.append((str(ours), str(theirs)))
    assert not different, different[:10]
    sn, sm, st, sp = sympy.symbols('n m theta phi')
    for v, sv in [(t, st), (p, sp)]:
        assert same(Ynm(n, m, t, p).diff(v), sympy.Ynm(sn, sm, st, sp).diff(sv)), v
    assert same(conjugate(Ynm(n, m, t, p)), sympy.conjugate(sympy.Ynm(sn, sm, st, sp)))
    value = complex(sympy_value(Ynm(Integer(2), Integer(1), Rational(1, 3), Rational(1, 5)).evalf(20)))
    assert abs(value - complex(sympy.N(sympy.Ynm(2, 1, sympy.Rational(1, 3), sympy.Rational(1, 5)), 20))) < 1e-14


def test_tensor_functions_match_sympy() -> None:
    different = []
    values: list[Expr] = [Integer(0), Integer(1), Integer(2), i, j, k, i + 1, 2*i]
    for a, b in itertools.product(values, repeat=2):
        if not same(KroneckerDelta(a, b), sympy.KroneckerDelta(sympy_value(a), sympy_value(b))):
            different.append((str(KroneckerDelta(a, b)), str(sympy.KroneckerDelta(sympy_value(a), sympy_value(b)))))
    for args in itertools.chain(itertools.product([Integer(1), Integer(2), Integer(3)], repeat=3),
                                itertools.product([Integer(1), i, j, k], repeat=3), [(Integer(1), Integer(2), Integer(3), Integer(4))]):
        ours, theirs = LeviCivita(*args), sympy.LeviCivita(*[sympy_value(a) for a in args])
        if not same(ours, theirs):
            different.append((str(ours), str(theirs)))
    assert not different, different[:10]
    assert KroneckerDelta(i, j).is_integer is True and LeviCivita(i, j, k).is_integer is True
