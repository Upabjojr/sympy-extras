"""The Rust port of ``galoistools`` and ``factortools`` against the Python
package: the docstring examples on both implementations, and the
factorisations of random products compared."""

from __future__ import annotations

import importlib
import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def test_galoistools_known(impl: str) -> None:
    F = mod(impl, "domains").GF(5)
    g = mod(impl, "galoistools")
    assert g.gf_factor([1, 0, 0, 0, 4], F) == (1, [([1, 1], 1), ([1, 2], 1), ([1, 3], 1), ([1, 4], 1)])
    assert g.gf_factor([3, 0, 0, 3], F) == (3, [([1, 1], 1), ([1, 4, 1], 1)])
    assert g.gf_factor([1, 3, 3, 1], F) == (1, [([1, 1], 3)])
    assert g.gf_ddf([1, 0, 0, 0, 4], F) == [([1, 0, 0, 0, 4], 1)]
    assert (g.gf_irreducible_p([1, 0, 2], F), g.gf_irreducible_p([1, 0, 1], F)) == (True, False)


def test_factortools_known(impl: str) -> None:
    d, f = mod(impl, "domains"), mod(impl, "factortools")
    ZZ, QQ = d.ZZ, d.QQ
    assert f.dup_zz_factor([2, 0, -8], ZZ) == (2, [([1, -2], 1), ([1, 2], 1)])
    assert f.dup_zz_factor([1, 0, 0, 0, 1], ZZ) == (1, [([1, 0, 0, 0, 1], 1)])
    assert f.dup_zz_factor([-1, 0, 2, 0, -1], ZZ) == (-1, [([1, -1], 2), ([1, 1], 2)])
    assert f.dup_qq_factor([Fraction(1, 2), Fraction(0), Fraction(-2)], QQ) == (
        Fraction(1, 2), [([Fraction(1), Fraction(-2)], 1), ([Fraction(1), Fraction(2)], 1)])
    assert ZZ.factor_dense([2, 0, -8]) == (2, [([1, -2], 1), ([1, 2], 1)])


def random_product(rng: random.Random, coefficient: Any, max_factors: int = 4) -> list[Any]:
    from sympy_extras.polys.densearith import dup_mul, dup_pow
    from sympy_extras.polys.domains import ZZ
    f: list[Any] = [coefficient(rng)] if rng.random() < 0.5 else [1]
    while not f[0]:
        f = [coefficient(rng)]
    for _ in range(rng.randint(1, max_factors)):
        degree = rng.randint(1, 4)
        g = [coefficient(rng) for _ in range(degree + 1)]
        while not g[0]:
            g[0] = coefficient(rng)
        f = dup_mul(f, dup_pow(g, rng.choice([1, 1, 1, 2, 3]), ZZ), ZZ)
    if rng.random() < 0.2:
        f = f + [0] * rng.randint(1, 3)
    return f


@pytest.mark.parametrize("seed", range(20))
def test_zz_factor_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.polys.domains import QQ, ZZ
    from sympy_extras.polys.factortools import dup_qq_factor, dup_zz_factor, dup_zz_irreducible_p
    rs, rd = load("sympy_extras_rs.polys.factortools"), load("sympy_extras_rs.polys.domains")
    rng = random.Random(seed)
    for _ in range(10):
        f = random_product(rng, lambda r: r.randint(-6, 6))
        assert rs.dup_zz_factor(f, rd.ZZ) == dup_zz_factor(f, ZZ), f
        assert rs.dup_zz_irreducible_p(f, rd.ZZ) == dup_zz_irreducible_p(f, ZZ), f
        q = [Fraction(c, rng.randint(1, 4)) for c in f]
        assert rs.dup_qq_factor(q, rd.QQ) == dup_qq_factor(q, QQ), q


@pytest.mark.parametrize("p", [2, 3, 7, 101, 65537])
@pytest.mark.parametrize("seed", range(8))
def test_gf_factor_agrees(rust: Any, p: int, seed: int) -> None:
    from sympy_extras.polys.densebasic import dup_strip
    from sympy_extras.polys.domains import GF
    from sympy_extras.polys import galoistools as gt
    rs, rd = load("sympy_extras_rs.polys.galoistools"), load("sympy_extras_rs.polys.domains")
    K, KR = GF(p), rd.GF(p)
    rng = random.Random(seed)
    for _ in range(6):
        f = dup_strip([c % p for c in random_product(rng, lambda r: r.randrange(p), 5)], K)
        assert rs.gf_factor(f, KR) == gt.gf_factor(f, K), f
        assert rs.gf_irreducible_p(f, KR) == gt.gf_irreducible_p(f, K), f
        if f and len(f) > 1:
            from sympy_extras.polys.sqfreetools import dup_sqf_part
            g = dup_sqf_part(f, K)
            assert rs.gf_factor_sqf(g, KR) == gt.gf_factor_sqf(g, K), g
            assert rs.gf_ddf(g, KR) == gt.gf_ddf(g, K), g
            for h, degree in gt.gf_ddf(g, K):
                assert rs.gf_edf(h, degree, KR, random.Random(seed)) == gt.gf_edf(h, degree, K, random.Random(seed))
            n = rng.randint(0, 3 * p)
            assert rs.gf_pow_mod(f, n, g, KR) == gt.gf_pow_mod(f, n, g, K)


@pytest.mark.parametrize("p", [2, 5, 13])
def test_gf_irreducible(rust: Any, p: int) -> None:
    from sympy_extras.polys.domains import GF
    from sympy_extras.polys.galoistools import gf_irreducible_p
    rs, rd = load("sympy_extras_rs.polys.galoistools"), load("sympy_extras_rs.polys.domains")
    rng = random.Random(p)
    for n in range(1, 8):
        f = rs.gf_irreducible(n, rd.GF(p), rng)
        assert len(f) == n + 1 and f[0] == 1 and gf_irreducible_p(f, GF(p)), f
        assert len(rs.gf_random(n, rd.GF(p), rng)) == n + 1


def test_hensel_and_cyclotomic(rust: Any) -> None:
    from sympy_extras.polys import factortools as ft
    from sympy_extras.polys.domains import ZZ
    rs, rd = load("sympy_extras_rs.polys.factortools"), load("sympy_extras_rs.polys.domains")
    for n in range(1, 40):
        assert rs.dup_zz_cyclotomic_poly(n) == ft.dup_zz_cyclotomic_poly(n)
        for sign in (1, -1):
            f = [1] + [0] * (n - 1) + [sign]
            assert rs.dup_zz_cyclotomic_factor(f) == ft.dup_zz_cyclotomic_factor(f)
    assert rs.dup_zz_cyclotomic_factor([2, 0, 1]) is None
    f = [1, 0, -10, 0, 1]
    assert rs.dup_zz_mignotte_bound(f, rd.ZZ) == ft.dup_zz_mignotte_bound(f, ZZ)
    # x**4 - 10 x**2 + 1 is irreducible over ZZ but splits modulo every prime
    from sympy_extras.polys.galoistools import gf_factor_sqf
    from sympy_extras.polys.domains import GF
    _, modular = gf_factor_sqf([c % 7 for c in f], GF(7))
    target = 2 * ft.dup_zz_mignotte_bound(f, ZZ) + 1
    assert rs.dup_zz_hensel_lift(7, f, modular, target) == ft.dup_zz_hensel_lift(7, f, modular, target)
    assert rs.dup_zz_zassenhaus(f) == ft.dup_zz_zassenhaus(f)
