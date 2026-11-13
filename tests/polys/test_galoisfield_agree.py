"""The Rust ports of ``domains.galoisfield`` (the fields ``GF(p**n)``) and
``galoisfactor`` (factorisation over them) against the Python modules:
the docstring examples on both implementations, the field operations on
random elements of random fields, and the factorisations of random
products with repeated factors (square-free and distinct-degree parts,
irreducibility)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed


def field_class(impl: str) -> Any:
    return load(impl + ".polys.domains.galoisfield").GaloisField


def gq(impl: str) -> Any:
    return load(impl + ".polys.galoisfactor")


def test_known(impl: str) -> None:
    GaloisField = field_class(impl)
    K = GaloisField(2, 3)
    assert (repr(K), K.modulus, K.order) == ("GF(2**3)", [1, 0, 1, 1], 8)
    a = K.generator
    assert (K.mul(a, a), K.pow(a, 3), K.pow(a, 7)) == ([1, 0, 0], [1, 1], [1])
    assert K.mul(K.invert([1, 1, 0]), [1, 1, 0]) == [1]
    assert len(list(K.elements())) == 8
    F = GaloisField(3, 2, [1, 0, 1])
    assert (F.pow(F.generator, 2), F.to_str([2, 1])) == ([2], "2*a + 1")
    with pytest.raises(ValueError):
        GaloisField(2, 2, [1, 0, 1])
    with pytest.raises(ValueError):
        GaloisField(2, 0)
    with pytest.raises(ZeroDivisionError):
        K.invert([])
    m = gq(impl)
    K4 = GaloisField(2, 2)
    f = [[1], [], [1], [1]]
    assert m.gfq_irreducible_p(f, K4) and m.gfq_factor(f, K4) == ([1], [([[1], [], [1], [1]], 1)])
    assert m.gfq_factor([[1], [1], [1]], K4) == ([1], [([[1], [1, 0]], 1), ([[1], [1, 1]], 1)])
    assert m.gfq_factor([[1], [], [], [], [1]], K4) == ([1], [([[1], [1]], 4)])
    assert m.gfq_sqf_list([[1], [], [1], []], K4) == ([1], [([[1], []], 1), ([[1], [1]], 2)])
    assert m.gfq_factor([[1], [], [1]], F) == ([1], [([[1], [1, 0]], 1), ([[1], [2, 0]], 1)])
    assert m.gfq_factor([[2]] + [[]]*8 + [[1]], F) == ([2], [([[1], [2]], 9)])


def random_field(rng: random.Random, impl: str) -> Any:
    p = rng.choice([2, 2, 3, 5, 7])
    n = rng.randint(1, 4 if p < 5 else 2)
    GaloisField = field_class(impl)
    if n > 1 and rng.random() < 0.5:
        # another irreducible modulus: a monic irreducible from the Python galoistools
        from sympy_extras.polys.domains import GF
        from sympy_extras.polys.galoistools import gf_irreducible_p
        while True:
            modulus = [1] + [rng.randrange(p) for _ in range(n)]
            if gf_irreducible_p(modulus, GF(p)):
                return GaloisField(p, n, modulus)
    return GaloisField(p, n)


def random_element(rng: random.Random, K: Any) -> list[int]:
    digits = [rng.randrange(K.p) for _ in range(K.degree)]
    while digits and not digits[0]:
        digits = digits[1:]
    return digits


@pytest.mark.parametrize("seed", range(20))
def test_field_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("galois field", seed))
    for _ in range(3):
        state = rng.getstate()
        results = []
        for impl in IMPLEMENTATIONS:
            rng.setstate(state)
            K = random_field(rng, impl)
            outcome: list[Any] = [repr(K), K.modulus, K.order, K.generator]
            for _ in range(10):
                a, b = random_element(rng, K), random_element(rng, K)
                outcome.append((K.add(a, b), K.sub(a, b), K.neg(a), K.mul(a, b), K.pow(a, rng.randint(0, 20)),
                                K.frobenius(a), K.pth_root(a), K.trace(a), K.to_str(a), K.canonical_unit(a),
                                K.gcdex(a, b), K.from_int(rng.randint(-20, 20))))
                if b:
                    outcome.append((K.invert(b), K.exquo(a, b), K.pow(b, -rng.randint(1, 5))))
            if K.order <= 64:
                outcome.append(list(K.elements()))
            results.append(outcome)
        assert results[0] == results[1]


def random_poly(rng: random.Random, K: Any, degree: int) -> list[list[int]]:
    f = [random_element(rng, K) for _ in range(degree + 1)]
    while f and not f[0]:
        f = f[1:]
    return f


@pytest.mark.parametrize("seed", range(20))
def test_factor_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("galois factor", seed))
    for _ in range(3):
        state = rng.getstate()
        results = []
        for impl in IMPLEMENTATIONS:
            rng.setstate(state)
            from sympy_extras.polys.densearith import dup_mul
            K = random_field(rng, impl)
            Kp = field_class("sympy_extras")(K.p, K.degree, K.modulus)
            f: list[list[int]] = [K.one]
            for _ in range(rng.randint(1, 3)):
                g = random_poly(rng, K, rng.randint(1, 4))
                for _ in range(rng.choice([1, 1, 2, K.p])):
                    f = dup_mul(f, g, Kp)
            m = gq(impl)
            lc, factors = m.gfq_factor(f, K)
            squarefree = [g for g, k in m.gfq_sqf_list(f, K)[1]]
            results.append(((lc, factors), m.gfq_sqf_list(f, K), [m.gfq_ddf(g, K) for g in squarefree],
                            m.gfq_irreducible_p(f, K), [m.gfq_irreducible_p(g, K) for g, _ in factors]))
        assert results[0] == results[1]
        assert all(results[0][4])
