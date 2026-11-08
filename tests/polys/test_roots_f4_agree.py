"""The Rust port of ``rootisolation`` and ``f4`` against the Python
package: the docstring examples on both implementations, isolating
intervals and root counts of random products, and reduced Gröbner bases of
random systems under the three orders over QQ and GF(p)."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed

F = Fraction


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def test_rootisolation_known(impl: str) -> None:
    r = mod(impl, "rootisolation")
    f = [F(1), F(0), F(-2)]
    assert r.dup_isolate_real_roots(f) == [((F(-2), F(0)), 1), ((F(0), F(2)), 1)]
    assert (r.dup_count_real_roots(f), r.dup_count_real_roots(f, F(0), F(3))) == (2, 1)
    assert r.dup_refine_real_root(f, F(0), F(2), F(1, 100)) == (F(181, 128), F(91, 64))
    assert r.dup_isolate_real_roots([F(1), F(-3), F(3), F(-1)]) == [((F(0), F(2)), 3)]
    d = mod(impl, "domains")
    assert d.ZZ.isolate_real_roots_dense([1, 0, -2]) == [((F(-2), F(0)), 1), ((F(0), F(2)), 1)]
    assert d.QQ.count_real_roots_dense(f, None, None) == 2


def test_f4_known(impl: str) -> None:
    d, rings, f4 = mod(impl, "domains"), mod(impl, "rings"), mod(impl, "f4")
    R = rings.PolyRing(d.QQ, ['x', 'y', 'z'], rings.grevlex)
    x, y, z = R.gens
    assert [str(g) for g in f4.f4([x*x + y*y + z*z - R.one, x + y + z, x*y*z - R.one], R)] == \
        ['z**3 - 1/2*z - 1', 'y**2 + y*z + z**2 - 1/2', 'x + y + z']
    L = rings.PolyRing(d.QQ, ['x', 'y'], rings.lex)
    X, Y = L.gens
    assert [str(g) for g in f4.f4([X*X - Y, Y*Y - X], L)] == ['x - y**2', 'y**4 - y']
    assert f4.f4([], L) == [] and [str(g) for g in f4.f4([X, L.from_int(3)], L)] == ['1']
    G = rings.PolyRing(d.GF(7), ['x', 'y'], rings.lex)
    X, Y = G.gens
    assert [str(g) for g in f4.f4([X*X - Y, (Y*Y).mul_ground(3) - X], G)] == ['x + 4*y**2', 'y**4 + 3*y']
    assert (f4.is_graded(R), f4.is_graded(L)) == (True, False)
    assert f4.row_echelon([{0: F(2), 1: F(4)}, {0: F(1), 2: F(1)}], d.QQ) == [{0: 1, 2: 1}, {1: 1, 2: F(-1, 2)}]
    with pytest.raises(ValueError):
        f4.f4([X], rings.PolyRing(d.ZZ, ['x', 'y'], rings.lex))


def random_real_poly(rng: random.Random) -> list[Fraction]:
    from sympy_extras.polys.densearith import dup_mul
    from sympy_extras.polys.domains import QQ
    f: list[Fraction] = [F(rng.randint(1, 3), rng.randint(1, 2))]
    for _ in range(rng.randint(1, 5)):
        kind = rng.random()
        if kind < 0.4:      # a rational root
            g = [F(rng.randint(1, 3)), F(rng.randint(-6, 6), rng.randint(1, 3))]
        elif kind < 0.8:    # two real or two complex roots
            g = [F(1), F(rng.randint(-4, 4)), F(rng.randint(-5, 5), rng.randint(1, 2))]
        else:               # a repeated factor
            h = [F(1), F(rng.randint(-3, 3))]
            g = dup_mul(h, h, QQ)
        f = dup_mul(f, g, QQ)
    return f


@pytest.mark.parametrize("seed", range(25))
def test_rootisolation_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.polys import rootisolation as py
    rs = mod("sympy_extras_rs", "rootisolation")
    rng = random.Random(seed)
    for _ in range(6):
        f = random_real_poly(rng)
        eps = F(1, rng.choice([10, 1000, 2**20]))
        a, b = sorted([F(rng.randint(-20, 20), rng.randint(1, 4)), F(rng.randint(-20, 20), rng.randint(1, 4))])
        assert rs.dup_isolate_real_roots(f) == py.dup_isolate_real_roots(f), f
        assert rs.dup_isolate_real_roots(f, eps) == py.dup_isolate_real_roots(f, eps), f
        assert rs.dup_count_real_roots(f, a, b) == py.dup_count_real_roots(f, a, b), (f, a, b)
        assert rs.dup_count_real_roots(f, None, b) == py.dup_count_real_roots(f, None, b), f
        assert rs.dup_sturm(f) == py.dup_sturm(f), f
        assert rs.dup_cauchy_bound(f) == py.dup_cauchy_bound(f), f
        assert rs.dup_fujiwara_bound(f) == py.dup_fujiwara_bound(f), f
        assert rs.dup_sign_at(f, a) == py.dup_sign_at(f, a), f
        from sympy_extras.polys.sqfreetools import dup_sqf_part
        from sympy_extras.polys.domains import QQ
        g = dup_sqf_part(f, QQ)
        assert rs.dup_isolate_real_roots_sqf(g, eps) == py.dup_isolate_real_roots_sqf(g, eps), g


def random_system(rng: random.Random, R: Any, coefficient: Any) -> list[Any]:
    gens = R.gens
    polys = []
    for _ in range(rng.randint(1, 3)):
        p = R.zero
        for _ in range(rng.randint(1, 4)):
            m = R.one
            for g in gens:
                m = m * g**rng.randint(0, 2 if len(gens) < 3 else 1)
            p = p + m.mul_ground(coefficient(rng))
        polys.append(p)
    return polys


DOMAINS = ["QQ", "GF(7)", "GF(32003)"]


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
@pytest.mark.parametrize("name", DOMAINS)
@pytest.mark.parametrize("nvars", [2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_f4_agrees(rust: Any, order: str, name: str, nvars: int, seed: int) -> None:
    rng = random.Random(stable_seed(order, name, nvars, seed))
    if name == "QQ":
        def coefficient(r: random.Random) -> Any:
            return F(r.randint(-3, 3) or 1, r.randint(1, 2))
    else:
        p = int(name[3:-1])

        def coefficient(r: random.Random) -> Any:
            return r.randrange(1, p)
    names = ['x', 'y', 'z'][:nvars]
    for _ in range(3):
        state = rng.getstate()
        results = []
        for impl in ("sympy_extras", "sympy_extras_rs"):
            rng.setstate(state)
            d, rings, f4 = mod(impl, "domains"), mod(impl, "rings"), mod(impl, "f4")
            K = d.GF(int(name[3:-1])) if name.startswith("GF(") else d.QQ
            R = rings.PolyRing(K, names, getattr(rings, order))
            system = random_system(rng, R, coefficient)
            basis = f4.f4(system, R)
            results.append(([dict(g.terms) for g in basis], [str(g) for g in basis],
                            [dict(g.terms) for g in f4.reduced_basis(system, R)]))
        assert results[0] == results[1], [str(p) for p in system]


@pytest.mark.parametrize("name", DOMAINS)
@pytest.mark.parametrize("seed", range(10))
def test_row_echelon_agrees(rust: Any, name: str, seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(5):
        rows: list[dict[int, Any]] = []
        for _ in range(rng.randint(1, 8)):
            row = {}
            for c in rng.sample(range(10), rng.randint(1, 6)):
                v = F(rng.randint(-5, 5), rng.randint(1, 3)) if name == "QQ" else rng.randrange(int(name[3:-1]))
                if v:
                    row[c] = v
            if row:
                rows.append(row)
        results = []
        for impl in ("sympy_extras", "sympy_extras_rs"):
            d = mod(impl, "domains")
            K = d.GF(int(name[3:-1])) if name.startswith("GF(") else d.QQ
            results.append(mod(impl, "f4").row_echelon([dict(r) for r in rows], K))
        assert results[0] == results[1], rows
