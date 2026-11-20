"""The Rust port of ``integrals/slater.py`` against the Python one: the
Mellin-Barnes G-functions of gamma quotients, Slater's expansion and the
line conditions (the numbers of the dummies are not compared)."""
from __future__ import annotations

import re
from typing import Any, Callable

import pytest

from sympy_extras.core import Integer, Rational, Symbol, Tuple, oo, pi, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.functions.special.hyper import meijerg
from sympy_extras.integrals.mellin import GammaQuotient
from sympy_extras.integrals.slater import MeijerG, expand_meijerg, line_conditions, mellin_barnes, slater_expansion

from ..conftest import load

rc = load("sympy_extras_rs.core")
z = Symbol('z', positive=True)
a, b = Symbol('a', positive=True), Symbol('b', positive=True)
nu = Symbol('nu')
H = Rational(1, 2)


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else re.sub(r'\b_?([a-zA-Z]+)\d+\b', r'\1', str(r))


def gargs(g: MeijerG) -> list[Basic]:
    return [g.prefactor, Tuple(*g.an), Tuple(*g.ap), Tuple(*g.bm), Tuple(*g.bq), g.z]


G_FUNCTIONS: list[MeijerG] = [
    MeijerG(1, [1], [], [1], [], Rational(1, 3)),
    MeijerG(1, [], [], [H], [0], z),
    MeijerG(1, [], [a, b], [], [0], z),
    MeijerG(1 / b, [H, 1], [], [1], [H], b**2 / a**2),
    MeijerG(1 / sqrt(pi), [1], [1], [H, 1], [], Rational(1)),
    MeijerG(1, [1 - a], [], [0], [], z),
    MeijerG(2 / sqrt(pi), [H], [], [H, 1], [], Rational(1, 64)),
    MeijerG(1, [0], [1], [0, 0], [], z),
    MeijerG(1, [], [], [0, H], [], z),
    MeijerG(1, [], [], [-Rational(1, 3), Rational(1, 3)], [], z),
]

QUOTIENTS: list[GammaQuotient] = [
    GammaQuotient(1, [(3, -1)], [(0, Rational(1)), (1, -Rational(1))], [], 0, 1),
    GammaQuotient(1, [(4, -1)], [(H, H), (1, -Rational(1))], [], -1, 1),
    GammaQuotient(1, [(2, 1)], [(0, H), (1, -Rational(1))], [(1, -H)], 0, 1),
    GammaQuotient(1, [(a, -1), (b, 1)], [(nu / 2, H), (1, -Rational(1))], [(1 + nu / 2, -H)], 0, 1),
]


@pytest.mark.parametrize('i', range(len(G_FUNCTIONS)))
def test_expand_meijerg(i: int) -> None:
    g = G_FUNCTIONS[i]

    def python() -> object:
        found = expand_meijerg(g)
        return None if found is None else Tuple(found.value, found.condition)
    got = outcome(lambda: rc.logic_call('slater', [Symbol('expand')] + gargs(g) + [Tuple()]))
    if got != 'SKIP':
        assert got == outcome(python), g


@pytest.mark.parametrize('i', range(len(G_FUNCTIONS)))
def test_slater_expansion(i: int) -> None:
    g = G_FUNCTIONS[i]
    e = meijerg(list(g.an), list(g.ap), list(g.bm), list(g.bq), g.z)
    got = outcome(lambda: rc.logic_call('slater', [Symbol('expansion'), e, Tuple()]))
    if got != 'SKIP':
        assert got == outcome(lambda: slater_expansion(e)), e


@pytest.mark.parametrize('i', range(len(G_FUNCTIONS)))
@pytest.mark.parametrize('limits', [(Integer(0), Integer(1)), (Integer(-1), Integer(1)), (Integer(0), oo), (-oo, Integer(0))])
def test_line_conditions(i: int, limits: tuple[Any, Any]) -> None:
    g = G_FUNCTIONS[i]
    got = outcome(lambda: rc.logic_call('slater', [Symbol('line')] + gargs(g) + list(limits)))
    assert got == outcome(lambda: line_conditions(g, limits[0], limits[1])), (g, limits)


@pytest.mark.parametrize('i', range(len(QUOTIENTS)))
def test_mellin_barnes(i: int) -> None:
    q = QUOTIENTS[i]

    def pairs(v: Any) -> Basic:
        return Tuple(*[Tuple(x, y) for x, y in v])

    def python() -> object:
        g = mellin_barnes(q)
        return None if g is None else Tuple(g.prefactor, Tuple(*g.an), Tuple(*g.ap), Tuple(*g.bm), Tuple(*g.bq), g.z)
    got = outcome(lambda: rc.logic_call('slater', [Symbol('mellin_barnes'), q.constant, pairs(q.bases), pairs(q.numerator),
                                                   pairs(q.denominator), q.extra]))
    assert got == outcome(python), q
