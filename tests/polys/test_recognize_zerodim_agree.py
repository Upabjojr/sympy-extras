"""The Rust ports of ``numberfields.recognize`` (algebraic numbers from
approximations, by LLL) and ``zerodim`` (the exact solutions of
zero-dimensional systems) against the Python modules: the docstring
examples on both implementations, the recognition of random algebraic
numbers and rationals from their decimal approximations, and the
solutions of random systems (compared through their reprs)."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed

F = Fraction


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def test_known(impl: str) -> None:
    rc = mod(impl, "numberfields.recognize")
    assert repr(rc.recognize_algebraic(F(1618033988749895, 10**15))) == 'Algebraic([1, -1, -1], 1.618033989)'
    assert repr(rc.recognize_algebraic(F(3, 7))) == 'Algebraic([7, -3], 3/7)'
    assert repr(rc.recognize_algebraic(F(1414213562373095, 10**15), digits=15)) == 'Algebraic([1, 0, -2], 1.414213562)'
    assert repr(rc.recognize_algebraic(F(3146264369941972, 10**15), digits=15, max_degree=4)) == \
        'Algebraic([1, 0, -10, 0, 1], 3.14626437)'
    assert repr(rc.recognize_algebraic(F(-1, 2), F(866025403784439, 10**15), digits=15)) == \
        'Algebraic([1, 1, 1], -0.5 + 0.8660254038*I)'
    assert rc.recognize_algebraic(F(3141592653589793, 10**15), digits=15, max_degree=4) is None
    with pytest.raises(ValueError):
        rc.recognize_algebraic(F(1), digits=0)
    zd, rings, d, f4 = mod(impl, "zerodim"), mod(impl, "rings"), mod(impl, "domains"), mod(impl, "f4")
    R = rings.PolyRing(d.QQ, ['x', 'y'], rings.lex)
    x, y = R.gens
    assert repr(zd.solve_zero_dimensional([x*x + y*y - R.from_int(5), x*y - R.from_int(2)], R)) == (
        '[[Algebraic([1, 1], -1), Algebraic([1, 2], -2)], [Algebraic([1, 2], -2), Algebraic([1, 1], -1)], '
        '[Algebraic([1, -2], 2), Algebraic([1, -1], 1)], [Algebraic([1, -1], 1), Algebraic([1, -2], 2)]]')
    solutions = zd.solve_zero_dimensional([x*x + y*y - R.from_int(1), y - x*x], R)
    assert (len(solutions), sum(1 for s in solutions if s[0].is_real and s[1].is_real)) == (4, 2)
    assert repr([s for s in solutions if s[0].is_real and s[0] > 0]) == \
        '[[Algebraic([1, 0, 1, 0, -1], 0.7861513778), Algebraic([1, 1, -1], 0.6180339887)]]'
    G = f4.f4([x*x + y*y - R.from_int(5), x*y - R.from_int(2)], R)
    assert zd.univariate_in_ideal(G, R, 0) == [1, 0, -5, 0, 4]
    assert zd.solve_zero_dimensional([x + y, R.one], R) == []
    with pytest.raises(ValueError):
        zd.solve_zero_dimensional([x*y], R)


def outcome(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return repr(function(*args, **kwargs))
    except (ValueError, ArithmeticError) as e:
        return (type(e).__name__, str(e))


@pytest.mark.parametrize("seed", range(15))
def test_recognize_agrees(rust: Any, seed: int) -> None:
    import math
    rng = random.Random(stable_seed("recognize", seed))
    for _ in range(3):
        kind = rng.random()
        if kind < 0.3:
            re, im = F(rng.randint(-50, 50), rng.randint(1, 30)), F(0)
        else:
            value = sum(rng.choice([-1, 1, 2]) * math.sqrt(p) for p in rng.sample([2, 3, 5, 7], rng.randint(1, 2)))
            re, im = F(round(value * 10**12), 10**12), F(0)
            if kind > 0.8:
                im = F(round(math.sqrt(rng.choice([2, 3])) * 10**12), 10**12)
        digits, degree = rng.choice([10, 12]), rng.randint(2, 4)
        results = [outcome(mod(impl, "numberfields.recognize").recognize_algebraic, re, im, digits, degree)
                   for impl in IMPLEMENTATIONS]
        assert results[0] == results[1], (re, im, digits, degree)


@pytest.mark.parametrize("seed", range(10))
def test_zero_dimensional_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("zero dimensional", seed))
    n = rng.choice([1, 2, 2, 3])
    system = []
    for i in range(n):
        d = rng.randint(1, 2)
        terms = {tuple(d if k == i else 0 for k in range(n)): F(1)}
        for _ in range(rng.randint(1, 2)):
            m = [0]*n
            for _ in range(rng.randint(0, d - 1)):
                m[rng.randrange(n)] += 1
            terms[tuple(m)] = terms.get(tuple(m), F(0)) + F(rng.randint(-3, 3))
        system.append(terms)
    results = []
    for impl in IMPLEMENTATIONS:
        zd, rings, d_ = mod(impl, "zerodim"), mod(impl, "rings"), mod(impl, "domains")
        R = rings.PolyRing(d_.QQ, ['x', 'y', 'z'][:n], rings.lex)
        polys = [R.from_dict({m: c for m, c in t.items() if c}) for t in system]
        results.append(outcome(zd.solve_zero_dimensional, polys, R))
    assert results[0] == results[1], system
