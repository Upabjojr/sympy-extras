"""The Rust ports of ``modulargroebner`` and ``buchberger`` against the
Python modules (on the typed rings of both): the integer functions, the
docstring examples on both implementations, random systems over QQ and
ZZ, homogeneous or not, with the default primes and with small primes
(which meet unlucky ones), comparing the bases and the traces, and the
direct computations by F4 and by Buchberger's algorithm."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed

F = Fraction


def rs(name: str) -> Any:
    return load("sympy_extras_rs.polys." + name)


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def outcome(function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except ValueError:
        return ValueError


def test_integers_agree(rust: Any) -> None:
    from sympy_extras.polys import modulargroebner as py
    mg = rs("modulargroebner")
    rng = random.Random(stable_seed("reconstruction"))
    for _ in range(300):
        m = rng.choice([101, 2**31 - 1, 2**61 - 1, 10**40 + 1])
        a = rng.randrange(m)
        bound = rng.choice([None, rng.randint(0, 10)])
        assert outcome(mg.rational_reconstruction, a, m, bound) == outcome(py.rational_reconstruction, a, m, bound)
        n, d = rng.randint(-1000, 1000), rng.randint(1, 1000)
        if d % 101:
            a = n * pow(d, -1, 101) % 101
            assert mg.rational_reconstruction(a, 101) == py.rational_reconstruction(a, 101)
        b, p = rng.randrange(97), 97
        assert mg.chinese_remainder(a % m, m, b, p) == py.chinese_remainder(a % m, m, b, p)
    for args in [(3, 1), (3, 101, 8)]:
        with pytest.raises(ValueError):
            mg.rational_reconstruction(*args)
        with pytest.raises(ValueError):
            py.rational_reconstruction(*args)
    ours, theirs = mg.default_primes(), py.default_primes()
    assert [next(ours) for _ in range(3)] == [next(theirs) for _ in range(3)]


def test_known(impl: str) -> None:
    mg, rings, d = mod(impl, "modulargroebner"), mod(impl, "rings"), mod(impl, "domains")
    from sympy_extras.ntheory.generate import primerange
    R = rings.PolyRing(d.QQ, ["x", "y", "z"], rings.lex)
    x, y, z = R.gens
    F_ = [x*x + y*y + z*z - R.one, x*y - z.mul_ground(F(1, 3)), x + y - z.mul_ground(F(2))]
    assert [str(g) for g in mg.modular_groebner(F_, R)] == \
        ["x + y - 2*z", "y**2 - 2*y*z + 1/3*z", "z**2 - 2/15*z - 1/5"]
    L = rings.PolyRing(d.QQ, ["x", "y"], rings.lex)
    X, Y = L.gens
    trace = mg.ModularTrace()
    assert [str(g) for g in mg.modular_groebner([X*X + Y, X*X + Y.mul_ground(F(7))], L,
                                                primes=primerange(2, 100), trace=trace)] == ["x**2", "y"]
    assert (trace.unlucky, trace.lucky) == ([2, 3], [5, 7])
    with pytest.raises(ValueError):
        mg.modular_groebner([X*X + Y, X*X + Y.mul_ground(F(7))], L, primes=[2, 3])
    assert mg.modular_groebner([], L) == []
    G7 = rings.PolyRing(d.GF(7), ["x", "y"], rings.lex)
    with pytest.raises(ValueError, match="the modular algorithm needs QQ or ZZ"):
        mg.modular_groebner([G7.gens[0]], G7)
    assert str(mg.homogeneous_order(rings.lex, 2)) == "HomogeneousOrder(lex, 2)"
    Q = rings.PolyRing(d.QQ, ["x", "y"], rings.grevlex)
    x, y = Q.gens
    system = [x**2 + y**2 - Q.one, x*y - Q.from_int(2)]
    assert [str(g) for g in mg.direct_groebner(system, Q)] == ["y**3 + 2*x - y", "x**2 + y**2 - 1", "x*y - 2"]
    bb = mod(impl, "buchberger")
    assert [str(g) for g in bb.buchberger(system, Q)] == ["y**3 + 2*x - y", "x**2 + y**2 - 1", "x*y - 2"]
    P = rings.PolyRing(d.QQ, ["x", "y"], rings.lex)
    x, y = P.gens
    assert str(bb.s_polynomial(x**2 - y, x*y - P.one)) == "x - y**2"
    with pytest.raises(ValueError):
        bb.buchberger([P.gens[0]], rings.PolyRing(d.ZZ, ["x", "y"], rings.lex))


SYSTEM_ORDERS = ["lex", "grevlex", "grlex"]


def random_system(rng: random.Random, n: int, integer: bool) -> list[dict[tuple[int, ...], Fraction]]:
    homogeneous = rng.random() < 0.3
    system: list[dict[tuple[int, ...], Fraction]] = []
    for _ in range(rng.randint(2, 3)):
        degree = rng.randint(1, 3)
        terms: dict[tuple[int, ...], Fraction] = {}
        for _ in range(rng.randint(2, 4)):
            e = [rng.randint(0, degree) for _ in range(n)]
            if homogeneous:
                if sum(e) > degree:
                    continue
                e[0] += degree - sum(e)
            # coefficients with a common factor and multiples of small primes
            c = rng.choice([1, -1, 2, 3, -5, 6, 7, 10, 30]) * rng.choice([1, 1, 2, 3])
            terms[tuple(e)] = F(c, 1 if integer else rng.choice([1, 1, 2, 3, 7]))
        if terms:
            system.append(terms)
    return system


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("order", SYSTEM_ORDERS)
def test_modular_groebner_agrees(rust: Any, seed: int, order: str) -> None:
    from sympy_extras.ntheory.generate import primerange
    rng = random.Random(stable_seed("modular", seed, order))
    integer = rng.random() < 0.3
    n = rng.randint(2, 3)
    names = ["x", "y", "z"][:n]
    system = random_system(rng, n, integer)
    for primes in (None, "small"):
        outcomes = []
        for impl in IMPLEMENTATIONS:
            mg, rings, d = mod(impl, "modulargroebner"), mod(impl, "rings"), mod(impl, "domains")
            R = rings.PolyRing(d.ZZ if integer else d.QQ, names, rings.as_order(order))
            polys = [R.from_dict({m: (c.numerator if integer else c) for m, c in t.items()}) for t in system]
            trace = mg.ModularTrace()
            basis = mg.modular_groebner(polys, R, primes=primerange(2, 10**6) if primes else None, trace=trace)
            outcomes.append(([{m: F(c) for m, c in g.terms.items()} for g in basis],
                             (trace.homogenized, trace.used, trace.skipped, trace.unlucky, trace.lucky,
                              trace.failed_verifications)))
        assert outcomes[0] == outcomes[1], (system, order, primes)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("field", ["QQ", "GF(101)"])
def test_direct_groebner_agrees(rust: Any, seed: int, field: str) -> None:
    rng = random.Random(stable_seed("direct", seed, field))
    order = rng.choice(SYSTEM_ORDERS)
    n = rng.randint(2, 3)
    system = random_system(rng, n, False)
    results = []
    for impl in IMPLEMENTATIONS:
        mg, rings, d, bb = mod(impl, "modulargroebner"), mod(impl, "rings"), mod(impl, "domains"), mod(impl, "buchberger")
        K = d.QQ if field == "QQ" else d.GF(101)
        R = rings.PolyRing(K, ["x", "y", "z"][:n], rings.as_order(order))
        polys = [R.from_dict({m: (c if field == "QQ" else K.from_int(c.numerator)) for m, c in t.items()})
                 for t in system]
        f4_basis = [dict(g.terms) for g in mg.direct_groebner(polys, R)]
        results.append((f4_basis, [dict(g.terms) for g in bb.buchberger(polys, R)]))
    assert results[0] == results[1]
    assert results[0][0] == results[0][1]
