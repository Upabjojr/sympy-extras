"""The Rust ports of ``groebnerwalk``, ``hilbert`` and ``subresultants``
against the Python modules, which work on SymPy's rings, ``Poly`` and dense
representation: the docstring examples, then random monomial ideals,
random pairs of polynomials and random ideals walked between orders over
QQ and GF(p)."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed

F = Fraction
P = 32003


def rs(name: str) -> Any:
    return load("sympy_extras_rs.polys." + name)


# -- hilbert


def test_hilbert_known(rust: Any) -> None:
    h = rs("hilbert")
    assert h.minimal_generators([(2, 1), (1, 0), (0, 3), (1, 3)]) == [(1, 0), (0, 3)]
    assert h.hilbert_numerator([(2, 0), (0, 2)]) == [1, 0, -2, 0, 1]
    assert h.hilbert_numerator([(1, 1, 0), (0, 1, 1)]) == [1, -2, 0, 1]
    assert h.hilbert_numerator([]) == [1]
    assert h.hilbert_series([(2, 0), (1, 1), (0, 2)], 2) == ([2, 1], 0)
    assert h.hilbert_series([(2, 0, 0), (1, 1, 0), (0, 2, 0)], 3) == ([2, 1], 1)
    assert h.hilbert_series([(1, 0), (0, 1)], 2) == ([1], 0)


@pytest.mark.parametrize("seed", range(30))
def test_hilbert_agrees(rust: Any, seed: int) -> None:
    from sympy import Poly, Symbol, cancel
    from sympy_extras.polys import hilbert as py
    h = rs("hilbert")
    t = Symbol("t")
    rng = random.Random(stable_seed("hilbert", seed))
    n = rng.randint(1, 5)
    monomials = [tuple(rng.randint(0, 4) for _ in range(n)) for _ in range(rng.randint(0, 9))]
    if rng.random() < 0.5:  # pure powers make the quotient zero-dimensional
        monomials += [tuple(rng.randint(1, 5) if i == j else 0 for i in range(n)) for j in range(n)]
    assert h.minimal_generators(monomials) == py.minimal_generators(monomials)
    expected = py.hilbert_numerator(monomials, t)
    assert h.hilbert_numerator(monomials) == ([] if expected.is_zero else [int(c) for c in expected.all_coeffs()])
    numerator, exponent = h.hilbert_series(monomials, n)
    q = Poly(numerator, t).as_expr() if numerator else 0
    assert cancel(py.hilbert_series(monomials, n, t) - q/(1 - t)**exponent) == 0


# -- subresultants


def test_psc_known(rust: Any) -> None:
    sr, d = rs("subresultants"), rs("domains")
    assert sr.dup_psc([1, 0, 1], [1, 0, -1], d.ZZ) == [4, 0, 1]
    assert sr.dup_psc([1, 0, -1, 0], [1, 0, -1], d.ZZ) == [0, 0, 1]
    # dmp_psc(x**2*y + x, x + y) over ZZ[y]
    assert sr.dup_psc([[1, 0], [1], []], [[1], [1, 0]], rs("domains.polynomialring").PolynomialRing(d.ZZ, "y")) == [[1, 0, -1, 0], [1]]
    assert sr.dup_psc([], [1], d.ZZ) == []


@pytest.mark.parametrize("seed", range(30))
def test_psc_agrees(rust: Any, seed: int) -> None:
    from sympy.polys.domains import QQ as SQQ, ZZ as SZZ
    from sympy_extras.polys import subresultants as py
    sr, d = rs("subresultants"), rs("domains")
    rng = random.Random(stable_seed("psc", seed))

    def dup(n: int) -> list[int]:
        f = [rng.randint(-4, 4) for _ in range(n + 1)]
        f[0] = f[0] or 1
        return f

    def dmp(n: int, m: int) -> list[list[int]]:
        f = [[rng.randint(-3, 3) for _ in range(rng.randint(0, m) + 1)] for _ in range(n + 1)]
        f[0] = [rng.randint(1, 3)] + f[0][1:]
        return [c[next((i for i, a in enumerate(c) if a), len(c)):] for c in f]

    if rng.random() < 0.5:
        common = dup(rng.randint(0, 2))  # a common factor gives defective subresultants
        from sympy.polys.densearith import dup_mul
        f = dup_mul(dup(rng.randint(0, 4)), common, SZZ)
        g = dup_mul(dup(rng.randint(0, 4)), common, SZZ)
    else:
        f, g = dup(rng.randint(0, 6)), dup(rng.randint(0, 6))
    assert sr.dup_psc(f, g, d.ZZ) == [int(c) for c in py.dup_psc(f, g, SZZ)]
    fq = [F(c, rng.randint(1, 3)) for c in f]
    assert sr.dup_psc(fq, g, d.QQ) == list(py.dup_psc([SQQ(c.numerator, c.denominator) for c in fq],
                                                       [SQQ(c) for c in g], SQQ))
    f2, g2 = dmp(rng.randint(1, 3), 2), dmp(rng.randint(1, 3), 2)
    expected = [[int(a) for a in c] for c in py.dmp_psc(f2, g2, 1, SZZ)]
    assert sr.dup_psc(f2, g2, rs("domains.polynomialring").PolynomialRing(d.ZZ, "y")) == expected


# -- groebnerwalk


def test_walk_known(rust: Any) -> None:
    w, rings, d, f4 = rs("groebnerwalk"), rs("rings"), rs("domains"), rs("f4")
    R = rings.PolyRing(d.QQ, ["x", "y", "z"], rings.grevlex)
    x, y, z = R.gens
    G = f4.f4([x*z - y*y, x*x - y*z], R)
    for lift in ("normal_form", "cofactors"):
        walked = w.groebner_walk(G, R, "lex", check=True, lift=lift)
        assert [str(g) for g in walked] == ["x**2 - y*z", "x*y**2 - y*z**2", "x*z - y**2", "y**4 - y*z**3"]
        T = R.with_order("lex")
        assert [g.terms for g in w.groebner_walk(walked, T, "grevlex", lift=lift)] == [g.terms for g in G]
    L = rings.PolyRing(d.QQ, ["x", "y"], rings.lex)
    X, Y = L.gens
    fs = [X*X - Y, X*Y - L.one]
    H, C = w.extended_groebner(fs, L)
    for h, cs in zip(H, C):
        total = L.zero
        for c, f in zip(cs, fs):
            total = total + c*f
        assert total.terms == h.terms
    assert w.is_groebner(f4.f4(fs, L), L) and not w.is_groebner(fs, L)
    assert str(w.initial_form(X*X - Y, [1, F(3)])) == "-y"
    with pytest.raises(ValueError):
        w.groebner_walk(G, R, "lex", lift="other")


def _sympy_terms(g: Any, p: int | None) -> dict[tuple[int, ...], Any]:
    if p is None:
        return {m: F(int(c.numerator), int(c.denominator)) for m, c in g.terms()}
    return {m: int(c) % p for m, c in g.terms()}


ORDERS = [("grevlex", "lex"), ("lex", "grevlex"), ("grlex", "lex"), ("grevlex", "weight")]


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("orders", ORDERS, ids=[f"{a}-{b}" for a, b in ORDERS])
def test_walk_agrees(rust: Any, seed: int, orders: tuple[str, str]) -> None:
    from sympy import QQ as SQQ, GF as SGF, grevlex, grlex, lex, ring
    from sympy.polys.groebnertools import groebner
    from sympy_extras.polys import groebnerwalk as py
    from sympy_extras.polys.orderings import WeightOrder
    w, rings, d, f4 = rs("groebnerwalk"), rs("rings"), rs("domains"), rs("f4")
    rng = random.Random(stable_seed("walk", seed, orders))
    p = P if rng.random() < 0.3 else None
    n = rng.randint(2, 3)
    names = ["x", "y", "z"][:n]
    weights = [rng.randint(1, 3) for _ in range(n)]
    sympy_orders = {"lex": lex, "grlex": grlex, "grevlex": grevlex, "weight": WeightOrder(weights, "lex")}
    rust_orders = {"lex": rings.lex, "grlex": rings.grlex, "grevlex": rings.grevlex,
                   "weight": rings.MonomialOrder.weight(weights, "lex")}
    source, target = orders
    system: list[dict[tuple[int, ...], Fraction]] = []
    for _ in range(rng.randint(1, 3)):
        terms: dict[tuple[int, ...], Fraction] = {}
        for _ in range(rng.randint(2, 4)):
            m = tuple(rng.randint(0, 2) for _ in range(n))
            terms[m] = F(rng.choice([-3, -2, -1, 1, 2, 3]), 1 if p else rng.randint(1, 2))
        system.append(terms)
    S, *_ = ring(",".join(names), SGF(p) if p else SQQ, sympy_orders[source])
    K = d.GF(p) if p else d.QQ
    R = rings.PolyRing(K, names, rust_orders[source])

    def sympy_coeff(c: Fraction) -> Any:
        return S.domain(c.numerator) / S.domain(c.denominator)

    Gs = groebner([S.from_dict({m: sympy_coeff(c) for m, c in t.items()}) for t in system], S)
    Gr = f4.f4([R.from_dict({m: (c.numerator * pow(c.denominator, -1, p)) % p if p else c
                             for m, c in t.items()}) for t in system], R)
    assert [g.terms for g in Gr] == [_sympy_terms(g, p) for g in Gs]
    for lift in ("normal_form", "cofactors"):
        expected = py.groebner_walk(Gs, S, sympy_orders[target], lift=lift)
        walked = w.groebner_walk(Gr, R, rust_orders[target], check=True, lift=lift)
        assert [g.terms for g in walked] == [_sympy_terms(g, p) for g in expected]
