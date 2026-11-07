"""Factorisation over GF(p), ZZ and QQ against SymPy (the oracle) and the
defining identities."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy import Poly, factor_list, symbols
from sympy.polys.domains import ZZ as SZZ, QQ as SQQ

from sympy_extras.polys.densearith import dup_mul, dup_pow
from sympy_extras.polys.densebasic import dup_convert, dup_degree, dup_strip
from sympy_extras.polys.domains import GF, QQ, ZZ
from sympy_extras.polys.factortools import (dup_qq_factor, dup_zz_factor, dup_zz_hensel_lift, dup_zz_irreducible_p,
                                             dup_zz_mignotte_bound)
from sympy_extras.polys.galoistools import gf_ddf, gf_factor, gf_irreducible, gf_irreducible_p, gf_pow_mod

x = symbols('x')


def _sympy_zz(f: list[int]) -> Poly:
    return Poly(f, x, domain=SZZ)


def _from_sympy(p: Poly) -> list[int]:
    return dup_strip([int(c) for c in p.all_coeffs()], ZZ)


def _random_zz(rng: random.Random, degree: int, bound: int) -> list[int]:
    f = dup_strip([rng.randint(-bound, bound) for _ in range(degree + 1)], ZZ)
    return f if f else [1]


def test_gf_against_sympy() -> None:
    rng = random.Random(1)
    for p in (2, 3, 5, 7, 11, 101):
        F = GF(p)
        for _ in range(25):
            f = dup_strip([rng.randrange(p) for _ in range(rng.randint(2, 9))], F)
            if dup_degree(f) < 1:
                continue
            lc, factors = gf_factor(f, F)
            product = [lc]
            for g, k in factors:
                product = dup_mul(product, dup_pow(g, k, F), F)
                assert g[0] == 1 and gf_irreducible_p(g, F), (p, f, g)
            assert product == f, (p, f)
            expected = Poly(f, x, modulus=p).factor_list()[1]
            assert sorted((dup_degree(g), k) for g, k in factors) == sorted((q.degree(), k) for q, k in expected), (p, f)
            # the distinct-degree factorisation of the square-free part
            sqf = [g for g, k in factors if k == 1]
            if len(sqf) == len(factors):
                degrees = sorted({dup_degree(g) for g in sqf})
                assert [d for _, d in gf_ddf(f, F)] == degrees


def test_gf_irreducible_and_powers() -> None:
    rng = random.Random(2)
    F = GF(7)
    for n in range(1, 7):
        f = gf_irreducible(n, F, rng)
        assert dup_degree(f) == n and gf_irreducible_p(f, F)
        assert Poly(f, x, modulus=7).is_irreducible
    g = [1, 0, 1]
    assert gf_pow_mod([1, 0], 4, g, F) == [1]
    assert not gf_irreducible_p([1, 0, 1], GF(5))


def test_hensel_lift() -> None:
    # x**4 - 1 = (x - 1)(x + 1)(x**2 + 1); modulo 5: four linear factors
    f = [1, 0, 0, 0, -1]
    factors = [[1, 1], [1, 2], [1, 3], [1, 4]]
    lifted, m = dup_zz_hensel_lift(5, f, factors, 10**6)
    assert m >= 10**6
    product = [1]
    for g in lifted:
        product = dup_mul(product, g, ZZ)
    assert [c % m for c in product] == [c % m for c in f]
    # a non-monic polynomial: 6x**2 + 5x + 1 = (2x + 1)(3x + 1)
    f = [6, 5, 1]
    lifted, m = dup_zz_hensel_lift(5, f, [[1, 3], [1, 2]], 10**4)
    product = [6]
    for g in lifted:
        product = dup_mul(product, g, ZZ)
    assert [c % m for c in product] == [c % m for c in f]


def test_zz_against_sympy() -> None:
    rng = random.Random(3)
    for _ in range(120):
        f = [1]
        for _ in range(rng.randint(1, 4)):
            f = dup_mul(f, dup_pow(_random_zz(rng, rng.randint(1, 3), 5), rng.randint(1, 2), ZZ), ZZ)
        if rng.random() < 0.3:
            f = dup_mul(f, [rng.choice([-6, -2, 3, 4])], ZZ)
        content, factors = dup_zz_factor(f, ZZ)
        expected_content, expected = factor_list(_sympy_zz(f))
        assert content == int(expected_content), f
        assert factors == sorted([(_from_sympy(g), k) for g, k in expected], key=lambda item: (len(item[0]), item[0], item[1])), f
        product = [content]
        for g, k in factors:
            product = dup_mul(product, dup_pow(g, k, ZZ), ZZ)
        assert product == f


def test_zz_hand_cases() -> None:
    assert dup_zz_factor([1, 0, 0, 0, -1], ZZ) == (1, [([1, -1], 1), ([1, 1], 1), ([1, 0, 1], 1)])
    # a Swinnerton-Dyer polynomial: irreducible, many modular factors
    swinnerton = _from_sympy(Poly((x**2 - 2)**2 - 2*(x**2 - 2)*0 - 0, x).compose(Poly(x, x)))
    assert dup_zz_irreducible_p([1, 0, -10, 0, 1], ZZ)  # (x**2 - 2)**2 - 8 x**2 ... minimal polynomial of sqrt(2) + sqrt(3)
    assert not dup_zz_irreducible_p([1, 0, -4], ZZ)
    assert dup_zz_factor([], ZZ) == (0, [])
    assert dup_zz_factor([7], ZZ) == (7, [])
    assert dup_zz_factor([0, 0, 3, 0, 0], ZZ) == dup_zz_factor([3, 0, 0], ZZ) == (3, [([1, 0], 2)])
    assert dup_zz_mignotte_bound([1, 0, -2], ZZ) >= 2
    assert swinnerton == [1, 0, -4, 0, 4]
    # x**n - 1 for a few n, against SymPy
    for n in (6, 12, 15):
        f = [1] + [0]*(n - 1) + [-1]
        content, factors = dup_zz_factor(f, ZZ)
        expected = factor_list(_sympy_zz(f))[1]
        assert sorted(len(g) for g, _ in factors) == sorted(g.degree() + 1 for g, _ in expected)


def test_qq_against_sympy() -> None:
    rng = random.Random(4)
    for _ in range(40):
        f = [1]
        for _ in range(rng.randint(1, 3)):
            f = dup_mul(f, _random_zz(rng, rng.randint(1, 3), 4), ZZ)
        fq = [Fraction(c, rng.randint(1, 4)) for c in f]
        fq = dup_strip(fq, QQ)
        content, factors = dup_qq_factor(fq, QQ)
        product = [content]
        for g, k in factors:
            product = dup_mul(product, dup_pow(g, k, QQ), QQ)
        assert product == fq
        expected_content, expected = factor_list(Poly([SQQ(c.numerator, c.denominator) for c in fq], x, domain=SQQ))
        assert sorted(len(g) for g, _ in factors) == sorted(g.degree() + 1 for g, _ in expected)
        assert content == Fraction(int(expected_content.p), int(expected_content.q))
    assert dup_convert([1, -2], ZZ, QQ) == [Fraction(1), Fraction(-2)]
