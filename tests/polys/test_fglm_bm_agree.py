"""The Rust ports of ``fglm`` (change of order of zero-dimensional
ideals) and ``berlekampmassey`` (shortest linear recurrences) against the
Python modules: the docstring examples, random zero-dimensional systems
over ``QQ`` and ``GF(p)`` converted to ``lex`` and ``grlex``, and random
recurrent and non-recurrent sequences over ``QQ``, ``GF(p)`` and
``QQ_I``."""

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
    d, rings, f4, fglm = mod(impl, "domains"), mod(impl, "rings"), mod(impl, "f4"), mod(impl, "fglm")
    R = rings.PolyRing(d.QQ, ['x', 'y'], rings.grevlex)
    x, y = R.gens
    G = f4.f4([x*x + y*y - R.from_int(5), x*y - R.from_int(2)], R)
    assert [str(g) for g in fglm.fglm(G, R, rings.lex)] == ['x + 1/2*y**3 - 5/2*y', 'y**4 - 5*y**2 + 4']
    S = rings.PolyRing(d.GF(7), ['x', 'y'], rings.grevlex)
    x, y = S.gens
    assert [str(g) for g in fglm.fglm([y*y - x, x*x - S.one], S, rings.lex)] == ['x + 6*y**2', 'y**4 + 6']
    with pytest.raises(ValueError):
        fglm.fglm([x*y], S, rings.lex)
    L = rings.PolyRing(d.QQ, ['x', 'y', 'z'], rings.lex)
    x, y, z = L.gens
    one = L.one
    assert [str(g) for g in fglm.groebner_by_fglm([x + y + z - one, x*x + y*y + z*z - one,
                                                   x*x*x + y*y*y + z*z*z - one], L)] == [
        'x + y + z - 1', 'y**2 + y*z - y + z**2 - z', 'z**3 - z**2']
    P = rings.PolyRing(d.QQ, ['x', 'y'], rings.lex)
    x, y = P.gens
    assert fglm.is_zero_dimensional([x*x - y, y*y*y - P.one], P) and not fglm.is_zero_dimensional([x*y], P)
    bm = mod(impl, "berlekampmassey")
    fib = [F(v) for v in (0, 1, 1, 2, 3, 5, 8, 13)]
    assert bm.berlekamp_massey(fib, d.QQ) == [1, -1, -1]
    assert bm.berlekamp_massey([1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1], d.GF(2)) == [1, 0, 1, 1]
    assert bm.extend_sequence(fib, d.QQ, 11)[-3:] == [21, 34, 55]
    assert bm.extend_sequence([1, 1, 0, 1, 1, 0], d.GF(2), 9) == [1, 1, 0, 1, 1, 0, 1, 1, 0]
    assert bm.satisfies_recurrence([1, 2, 4, 3, 1], [1, 3], d.GF(5))


def zero_dimensional_system(rng: random.Random, R: Any, rational: bool) -> list[Any]:
    """``x_i**d_i`` plus random terms of lower total degree, for every
    variable: zero-dimensional, the pure powers leading for graded orders."""
    n = len(R.gens)
    system = []
    for i in range(n):
        d = rng.randint(1, 3)
        f = R.term(tuple(d if k == i else 0 for k in range(n)), R.domain.one)
        for _ in range(rng.randint(1, 3)):
            m = [0]*n
            for _ in range(rng.randint(0, d - 1)):
                m[rng.randrange(n)] += 1
            c: Any = rng.randint(-5, 5)
            c = F(c, rng.randint(1, 3)) if rational else R.domain.from_int(c)
            f = f + R.term(tuple(m), c)
        system.append(f)
    return system


@pytest.mark.parametrize("field", ["QQ", "GF(32003)"])
@pytest.mark.parametrize("seed", range(10))
def test_fglm_agrees(rust: Any, field: str, seed: int) -> None:
    rng = random.Random(stable_seed("fglm", field, seed))
    for _ in range(3):
        n = rng.choice([2, 2, 3])
        target = rng.choice(["lex", "lex", "grlex"])
        state = rng.getstate()
        results = []
        for impl in IMPLEMENTATIONS:
            rng.setstate(state)
            d, rings, f4, fglm = mod(impl, "domains"), mod(impl, "rings"), mod(impl, "f4"), mod(impl, "fglm")
            K = d.QQ if field == "QQ" else d.GF(32003)
            R = rings.PolyRing(K, ['x', 'y', 'z'][:n], rings.grevlex)
            system = zero_dimensional_system(rng, R, field == "QQ")
            basis = f4.f4(system, R)
            converted = fglm.fglm(basis, R, rings.as_order(target))
            T = rings.PolyRing(K, ['x', 'y', 'z'][:n], rings.as_order(target))
            direct = fglm.groebner_by_fglm([T.from_dict(dict(f.terms)) for f in system], T)
            results.append(([dict(g.terms) for g in converted], [dict(g.terms) for g in direct],
                            fglm.is_zero_dimensional(basis, R)))
        assert results[0] == results[1]


def random_sequence(rng: random.Random, impl: str, field: str) -> tuple[Any, list[Any]]:
    d = mod(impl, "domains")
    if field == "QQ_I":
        K = mod(impl, "domains.gaussiandomains").QQ_I
        G = mod(impl, "domains.gaussiandomains").GaussianRational

        def element() -> Any:
            return G(F(rng.randint(-3, 3)), F(rng.randint(-3, 3)))
    elif field == "QQ":
        K = d.QQ

        def element() -> Any:
            return F(rng.randint(-4, 4), rng.randint(1, 3))
    else:
        K = d.GF(7)

        def element() -> Any:
            return K.from_int(rng.randint(0, 6))
    length = rng.randint(0, 12)
    if rng.random() < 0.6:
        # a recurrence of order L, enough terms to find it
        order = rng.randint(1, 4)
        coefficients = [element() for _ in range(order)]
        sequence = [element() for _ in range(order)]
        while len(sequence) < max(length, 2*order):
            total = K.zero
            for i, c in enumerate(coefficients):
                total = K.sub(total, K.mul(c, sequence[-1 - i]))
            sequence.append(total)
        return K, sequence
    return K, [element() for _ in range(length)]


def plain(v: Any) -> Any:
    return (F(v.x), F(v.y)) if hasattr(v, "x") else F(int(v)) if not isinstance(v, F) else v


@pytest.mark.parametrize("field", ["QQ", "GF(7)", "QQ_I"])
@pytest.mark.parametrize("seed", range(10))
def test_berlekamp_massey_agrees(rust: Any, field: str, seed: int) -> None:
    rng = random.Random(stable_seed("berlekamp massey", field, seed))
    for _ in range(8):
        state = rng.getstate()
        results = []
        for impl in IMPLEMENTATIONS:
            rng.setstate(state)
            bm = mod(impl, "berlekampmassey")
            K, sequence = random_sequence(rng, impl, field)
            polynomial = bm.berlekamp_massey(sequence, K)
            results.append(([plain(c) for c in polynomial],
                            [plain(c) for c in bm.extend_sequence(sequence, K, len(sequence) + 4)],
                            bm.satisfies_recurrence(sequence, polynomial, K)))
        assert results[0] == results[1]
        assert results[0][2]
