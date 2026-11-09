"""The Rust port of ``modulargroebner`` against the Python module (on
SymPy's rings): the integer functions, the docstring examples, and random
systems over QQ and ZZ, homogeneous or not, with the default primes and
with small primes (which meet unlucky ones), comparing the bases and the
traces."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed

F = Fraction


def rs(name: str) -> Any:
    return load("sympy_extras_rs.polys." + name)


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


def test_known(rust: Any) -> None:
    from sympy import primerange
    mg, rings, d = rs("modulargroebner"), rs("rings"), rs("domains")
    R = rings.PolyRing(d.QQ, ["x", "y", "z"], rings.lex)
    x, y, z = R.gens
    F_ = [x*x + y*y + z*z - R.one, x*y - z.mul_ground(F(1, 3)), x + y - z.mul_ground(2)]
    assert [str(g) for g in mg.modular_groebner(F_, R)] == \
        ["x + y - 2*z", "y**2 - 2*y*z + 1/3*z", "z**2 - 2/15*z - 1/5"]
    L = rings.PolyRing(d.QQ, ["x", "y"], rings.lex)
    X, Y = L.gens
    trace = mg.ModularTrace()
    assert [str(g) for g in mg.modular_groebner([X*X + Y, X*X + Y.mul_ground(7)], L,
                                                primes=primerange(2, 100), trace=trace)] == ["x**2", "y"]
    assert (trace.unlucky, trace.lucky) == ([2, 3], [5, 7])
    with pytest.raises(ValueError):
        mg.modular_groebner([X*X + Y, X*X + Y.mul_ground(7)], L, primes=[2, 3])
    assert mg.modular_groebner([], L) == []


SYSTEM_ORDERS = ["lex", "grevlex", "grlex"]


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("order", SYSTEM_ORDERS)
def test_modular_groebner_agrees(rust: Any, seed: int, order: str) -> None:
    from sympy import QQ as SQQ, ZZ as SZZ, grevlex, grlex, lex, primerange, ring
    from sympy_extras.polys import modulargroebner as py
    mg, rings, d = rs("modulargroebner"), rs("rings"), rs("domains")
    rng = random.Random(stable_seed("modular", seed, order))
    integer = rng.random() < 0.3
    n = rng.randint(2, 3)
    names = ["x", "y", "z"][:n]
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
    sympy_order = {"lex": lex, "grevlex": grevlex, "grlex": grlex}[order]
    S, *_ = ring(",".join(names), SZZ if integer else SQQ, sympy_order)
    R = rings.PolyRing(d.ZZ if integer else d.QQ, names, getattr(rings, order))

    def sympy_poly(t: dict[tuple[int, ...], Fraction]) -> Any:
        return S.from_dict({m: S.domain.convert(c.numerator) / S.domain.convert(c.denominator)
                            if not integer else S.domain(c.numerator) for m, c in t.items()})

    sympy_system = [sympy_poly(t) for t in system]
    rust_system = [R.from_dict({m: (c.numerator if integer else c) for m, c in t.items()}) for t in system]

    def terms_of(g: Any) -> dict[tuple[int, ...], Fraction]:
        return {m: F(int(c.numerator), int(c.denominator)) if not integer else F(int(c)) for m, c in g.terms()}

    for primes in (None, "small"):
        py_trace, rs_trace = py.ModularTrace(), mg.ModularTrace()
        expected = py.modular_groebner(sympy_system, S, primes=primerange(2, 10**6) if primes else None,
                                       trace=py_trace)
        got = mg.modular_groebner(rust_system, R, primes=primerange(2, 10**6) if primes else None,
                                  trace=rs_trace)
        assert [{m: F(c) for m, c in g.terms.items()} for g in got] == [terms_of(g) for g in expected]
        assert (rs_trace.homogenized, rs_trace.used, rs_trace.skipped, rs_trace.unlucky, rs_trace.lucky,
                rs_trace.failed_verifications) == \
            (py_trace.homogenized, py_trace.used, py_trace.skipped, py_trace.unlucky, py_trace.lucky,
             py_trace.failed_verifications)
