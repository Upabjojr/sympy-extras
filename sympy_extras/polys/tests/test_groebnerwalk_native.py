"""Tests of the native Gröbner walk (:mod:`sympy_extras.polys.groebnerwalk`)
on the typed rings: every walked basis against the basis computed directly
by F4 in the target order; the SymPy face is tested in
``test_groebnerwalk.py``."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy_extras._testing import untyped
from sympy_extras.core import Integer, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.polys.domains import GF, QQ
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.f4 import f4
from sympy_extras.polys.groebnerwalk import extended_groebner, groebner_walk, initial_form, is_groebner, normal_form
from sympy_extras.polys.orderings import BlockOrder, MonomialOrder, WeightOrder, grevlex, grlex, lex
from sympy_extras.polys.rings import PolyRing

x, y, z, w = [Symbol(name) for name in 'xyzw']


def _check_walk(F: list[Expr], names: str, source: MonomialOrder, target: MonomialOrder,
                domain: Domain[Fraction] = QQ) -> list[str]:
    R = PolyRing(domain, names.split(','), source)
    T = R.with_order(target)
    G = f4([R.from_expr(f) for f in F], R)
    expected = f4([T.from_expr(f) for f in F], T)
    for lift in ('normal_form', 'cofactors'):
        walked = groebner_walk(G, R, target, check=True, lift=lift)
        assert walked == expected, (F, lift, walked, expected)
        assert all(g.ring == T for g in walked)
        assert is_groebner(walked, T)
    return [str(g) for g in expected]


def test_walk_against_f4() -> None:
    twisted = [x*z - y**2, x**2 - y*z]
    assert _check_walk(twisted, 'x,y,z', grevlex, lex) == ['x**2 - y*z', 'x*y**2 - y*z**2', 'x*z - y**2',
                                                            'y**4 - y*z**3']
    _check_walk(twisted, 'x,y,z', lex, grevlex)
    _check_walk(twisted, 'x,y,z', grlex, lex)
    _check_walk(twisted, 'x,y,z', grevlex, grlex)
    _check_walk(twisted, 'x,y,z', grevlex, WeightOrder((1, 2, 3), lex))
    _check_walk(twisted, 'x,y,z', WeightOrder((1, 2, 3), lex), grevlex)
    _check_walk([x**2*y - z**3, x*y*z - w**2, y**3 - x*w, x**3 - y*z*w], 'x,y,z,w', grevlex, lex)
    _check_walk([x**2 + y**2 + z**2 - 1, x*y - z, x - y**2], 'x,y,z', grevlex, lex)
    _check_walk([x**5 - y**3*z, x*y - z**2, y**4 - x**2*z], 'x,y,z', grevlex, lex)
    _check_walk([x**2 - y/2, x*y - 3], 'x,y', grevlex, lex)
    _check_walk([x**2 - y*z, y**2 - x*z], 'x,y,z', grevlex, WeightOrder((3, 1, 1), grevlex))


def test_walk_over_a_finite_field() -> None:
    R = PolyRing(GF(7), ['x', 'y'], grevlex)
    X, Y = R.gens
    G = f4([X**2 + Y**2 - R.one, X - Y], R)
    T = R.with_order(lex)
    assert groebner_walk(G, R, 'lex', check=True) == f4([T.from_expr(x**2 + y**2 - 1), T.from_expr(x - y)], T)


def test_random_walks() -> None:
    rng = random.Random(2)
    for _ in range(40):
        R = PolyRing(QQ, ['x', 'y', 'z'], grevlex)
        T = R.with_order(lex)
        F: list[Expr] = []
        for _ in range(rng.randint(1, 3)):
            f: Expr = Integer(0)
            for _ in range(rng.randint(2, 4)):
                f = f + rng.choice([-3, -1, 1, 2])*x**rng.randint(0, 2)*y**rng.randint(0, 2)*z**rng.randint(0, 1)
            F.append(f)
        polys = [R.from_expr(f) for f in F]
        if all(not p.terms for p in polys):
            continue
        G = f4(polys, R)
        for lift in ('normal_form', 'cofactors'):
            assert groebner_walk(G, R, lex, lift=lift) == f4([T.from_expr(f) for f in F], T), F


def test_trivial_cases_and_errors() -> None:
    R = PolyRing(QQ, ['x', 'y'], grevlex)
    X, Y = R.gens
    T = R.with_order(lex)
    assert groebner_walk([], R, lex) == [] and groebner_walk([R.zero], R, lex) == []
    assert groebner_walk([R.one], R, lex) == [T.one]
    assert groebner_walk([X**2 + R.one], R, lex) == [T.from_expr(x**2 + 1)]
    for bad in (lambda: groebner_walk([X], R, lex, lift='magic'),
                lambda: groebner_walk([X], R, BlockOrder([(grevlex, 1), (grevlex, 1)])),
                lambda: untyped(groebner_walk)([X], R, 'ilex')):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('ValueError expected')


def test_extended_groebner_and_normal_form() -> None:
    R = PolyRing(QQ, ['x', 'y'], lex)
    X, Y = R.gens
    for F in ([X**2 - Y, X*Y - R.one], [R.zero, X], [X**3 - Y**2, X**2*Y - X, Y**3 - R.one], [X + Y, X - Y]):
        H, C = extended_groebner(F, R)
        assert is_groebner(H, R)
        assert all(h == sum((c*f for c, f in zip(row, F)), R.zero) for h, row in zip(H, C))
        # the same ideal: the reduced bases agree
        assert f4(H, R) == f4(F, R)
    H, C = extended_groebner([R.zero, X], R)
    assert H == [X] and C == [[R.zero, R.one]]
    assert not is_groebner([X**2 - Y, X*Y - R.one], R) and is_groebner([], R)
    # x**3 + y = x*(x**2 - y) + x*y + y
    assert str(normal_form(X**3 + Y, [X**2 - Y])) == 'x*y + y'
    assert normal_form(X**2 - Y, f4([X**2 - Y, X*Y - R.one], R)) == R.zero


def test_initial_form() -> None:
    R = PolyRing(QQ, ['x', 'y'])
    X, Y = R.gens
    f = X**2*Y + X*Y**2 + X
    assert initial_form(f, (1, 2)) == X*Y**2
    assert initial_form(f, (1, 1)) == X**2*Y + X*Y**2
    assert initial_form(f, (Fraction(1, 2), 1)) == X*Y**2
    assert initial_form(R.zero, (1, 1)) == R.zero
    assert initial_form(X**2 - Y, [1, Fraction(3)]) == -Y
