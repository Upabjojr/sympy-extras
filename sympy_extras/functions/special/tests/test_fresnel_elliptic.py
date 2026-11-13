"""The typed Fresnel and elliptic integrals against SymPy (the oracle),
compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, zoo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import nan
from sympy_extras.functions import elliptic_e, elliptic_f, elliptic_k, elliptic_pi, fresnelc, fresnels

x, z, m, n = Symbol('x'), Symbol('z'), Symbol('m'), Symbol('n')
r = Symbol('r', real=True)


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_fresnel_matches_sympy() -> None:
    different = []
    for a in [Integer(0), Integer(1), oo, -oo, I*oo, nan, x, -x, I*x, -I*x, 2*I*x, r, -r, I, Rational(1, 2)]:
        sa = sympy_value(a)
        for ours, theirs in [(fresnels(a), sympy.fresnels(sa)), (fresnelc(a), sympy.fresnelc(sa))]:
            if not same(ours, theirs):
                different.append((str(ours), str(theirs)))
    assert not different, different


def test_elliptic_match_sympy() -> None:
    parameters: list[Expr] = [Integer(0), Integer(1), Integer(-1), Rational(1, 2), Integer(2), oo, -oo, I*oo, zoo, m, n]
    amplitudes: list[Expr] = [Integer(0), pi/2, pi, -pi, 3*pi/2, z, -z, Rational(1, 3)]
    different = []
    for a in parameters:
        sa = sympy_value(a)
        for ours, theirs in [(elliptic_k(a), sympy.elliptic_k(sa)), (elliptic_e(a), sympy.elliptic_e(sa))]:
            if not same(ours, theirs):
                different.append((str(ours), str(theirs)))
        for b in amplitudes:
            sb = sympy_value(b)
            for ours, theirs in [(elliptic_f(b, a), sympy.elliptic_f(sb, sa)), (elliptic_e(b, a), sympy.elliptic_e(sb, sa))]:
                if not same(ours, theirs):
                    different.append((str(ours), str(theirs)))
    for a, b in itertools.product(parameters, repeat=2):
        sa, sb = sympy_value(a), sympy_value(b)
        if not same(elliptic_pi(a, b), sympy.elliptic_pi(sa, sb)):
            different.append((str(elliptic_pi(a, b)), str(sympy.elliptic_pi(sa, sb))))
        for c in [Integer(0), pi/2, z, -z]:
            sc = sympy_value(c)
            if not same(elliptic_pi(a, c, b), sympy.elliptic_pi(sa, sc, sb)):
                different.append((str(elliptic_pi(a, c, b)), str(sympy.elliptic_pi(sa, sc, sb))))
    # zoo absorbs a numerical value the core knows to be finite
    # (elliptic_e(-1), by evaluation), where SymPy keeps the sum
    unexpected = [d for d in different if not (d[0] == 'zoo' and d[1].endswith('+ zoo'))]
    assert not unexpected and len(different) == 2, different[:20]


def test_derivatives_and_numbers() -> None:
    sx, sz, sm, sn = sympy.symbols('x z m n')
    cases: list[tuple[Expr, sympy.Expr, list[tuple[Symbol, sympy.Symbol]]]] = [
        (fresnels(x**2), sympy.fresnels(sx**2), [(x, sx)]), (fresnelc(x), sympy.fresnelc(sx), [(x, sx)]),
        (elliptic_k(m), sympy.elliptic_k(sm), [(m, sm)]), (elliptic_f(z, m), sympy.elliptic_f(sz, sm), [(z, sz), (m, sm)]),
        (elliptic_e(m), sympy.elliptic_e(sm), [(m, sm)]), (elliptic_e(z, m), sympy.elliptic_e(sz, sm), [(z, sz), (m, sm)]),
        (elliptic_pi(n, m), sympy.elliptic_pi(sn, sm), [(n, sn), (m, sm)]),
        (elliptic_pi(n, z, m), sympy.elliptic_pi(sn, sz, sm), [(n, sn), (z, sz), (m, sm)])]
    for ours, theirs, variables in cases:
        for v, sv in variables:
            assert same(ours.diff(v), theirs.diff(sv)), (ours, v)
    for e, s in [(fresnels(Rational(1, 3)), sympy.fresnels(sympy.Rational(1, 3))),
                 (elliptic_k(Rational(1, 3)), sympy.elliptic_k(sympy.Rational(1, 3))),
                 (elliptic_f(Integer(1), Rational(1, 3)), sympy.elliptic_f(1, sympy.Rational(1, 3))),
                 (elliptic_e(Integer(1), Rational(1, 3)), sympy.elliptic_e(1, sympy.Rational(1, 3))),
                 (elliptic_pi(Rational(1, 5), Integer(1), Rational(1, 3)),
                  sympy.elliptic_pi(sympy.Rational(1, 5), 1, sympy.Rational(1, 3)))]:
        assert abs(complex(sympy_value(e.evalf(20))) - complex(sympy.N(s, 20))) < 1e-13, e


def test_fresnel_asymptotic_series_are_native() -> None:
    # the series at oo and -oo went to SymPy through the bridge (the
    # classes had no _eval_aseries of their own); ported, checked against
    # SymPy's with no bridge registered
    from sympy_extras.core.bridge import Bridge, bridge, register_bridge
    X = sympy.Symbol('x')
    previous = bridge()
    register_bridge(Bridge())
    try:
        for f, F in ((fresnels, sympy.fresnels), (fresnelc, sympy.fresnelc)):
            for arg, theirs_arg, point, order in ((x, X, oo, 8), (2 * x, 2 * X, oo, 6), (x, X, -oo, 6),
                                                  (x**2, X**2, oo, 9)):
                ours = f(arg).series(x, point, order).removeO()
                register_bridge(previous)
                theirs = F(theirs_arg).series(X, sympy.oo if point == oo else -sympy.oo, order).removeO()
                assert sympy.expand(sympy_value(ours) - theirs) == 0, (f, arg, point)
                register_bridge(Bridge())
    finally:
        register_bridge(previous)
