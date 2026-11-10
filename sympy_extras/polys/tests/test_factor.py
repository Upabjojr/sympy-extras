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
from sympy_extras.polys.densearith import dup_mul_ground, dup_sub
from sympy_extras.polys.densetools import dup_primitive, dup_shift
from sympy_extras.polys.factortools import (dup_qq_factor, dup_zz_cyclotomic_factor, dup_zz_cyclotomic_poly,
                                             dup_zz_factor, dup_zz_hensel_lift, dup_zz_irreducible_p,
                                             dup_zz_mignotte_bound, dup_zz_van_hoeij, dup_zz_zassenhaus)
from sympy_extras.polys.sqfreetools import dup_sqf_p
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


def test_cyclotomic() -> None:
    from sympy import cyclotomic_poly
    for n in range(1, 40):
        assert dup_zz_cyclotomic_poly(n) == _from_sympy(Poly(cyclotomic_poly(n, x), x)), n
    for n in (1, 2, 6, 12, 30, 100):
        minus = [1] + [0]*(n - 1) + [-1]
        plus = [1] + [0]*(n - 1) + [1]
        for f in (minus, plus):
            found = dup_zz_cyclotomic_factor(f)
            assert found is not None
            product = [1]
            for g in found:
                product = dup_mul(product, g, ZZ)
            assert product == f
            assert sorted(len(g) for g in found) == sorted(g.degree() + 1 for g, _ in factor_list(_sympy_zz(f))[1])
    assert dup_zz_cyclotomic_factor([1, 0, -2]) is None and dup_zz_cyclotomic_factor([2, 0, -2]) is None


# ---------------------------------------------------------------------------
# van Hoeij's recombination

def _swinnerton_dyer(n: int) -> list[int]:
    """The minimal polynomial of ``sqrt(2) + sqrt(3) + ... + sqrt(p_n)``,
    of degree ``2**n``: ``f(x + s) f(x - s)`` with ``s**2 = p``, written
    ``A**2 - p B**2`` for ``f(x + s) = A + s B``."""
    f = [1, 0]
    for p in (2, 3, 5, 7, 11, 13, 17)[:n]:
        d = dup_degree(f)
        even = [0]*(d + 1)
        odd = [0]*(d + 1)
        for i, c in enumerate(reversed(f)):
            binomial = 1
            for j in range(i, -1, -1):
                # the term c binomial(i, j) x**j s**(i - j)
                term = c*binomial*p**((i - j)//2)
                if (i - j) % 2:
                    odd[j] += term
                else:
                    even[j] += term
                binomial = binomial*j//(i - j + 1)
        A = dup_strip(even[::-1], ZZ)
        B = dup_strip(odd[::-1], ZZ)
        f = dup_sub(dup_mul(A, A, ZZ), dup_mul_ground(dup_mul(B, B, ZZ), p, ZZ), ZZ)
    return f


def _sympy_factors(f: list[int]) -> list[list[int]]:
    return sorted((_from_sympy(g) for g, _ in _sympy_zz(f).factor_list()[1]), key=lambda g: (len(g), g))


def test_swinnerton_dyer_polynomials() -> None:
    # known: x**4 - 10 x**2 + 1 and x**8 - 40 x**6 + 352 x**4 - 960 x**2 + 576
    assert _swinnerton_dyer(2) == [1, 0, -10, 0, 1]
    assert _swinnerton_dyer(3) == [1, 0, -40, 0, 352, 0, -960, 0, 576]
    # irreducible, 2**(n - 1) quadratic factors modulo every prime: the
    # subset search of Zassenhaus needs 2**31 trial divisions for n = 6
    for n in (2, 3, 4, 5, 6):
        f = _swinnerton_dyer(n)
        assert dup_zz_van_hoeij(f) == [f]
    f = _swinnerton_dyer(6)
    assert dup_zz_zassenhaus(f) == [f]
    assert dup_zz_irreducible_p(f, ZZ)


def test_van_hoeij_products_of_swinnerton_dyer() -> None:
    s4 = _swinnerton_dyer(4)
    s3 = _swinnerton_dyer(3)
    parts = [s4, dup_shift(s4, 1, ZZ), dup_shift(s3, -2, ZZ)]
    f = dup_mul(dup_mul(parts[0], parts[1], ZZ), parts[2], ZZ)
    expected = sorted(parts, key=lambda g: (len(g), g))
    assert dup_zz_van_hoeij(f) == expected
    assert dup_zz_factor(f, ZZ) == (1, [(g, 1) for g in expected])
    # not monic: s5(2 x + 1) s5(x), with the content of s5(2 x + 1) removed
    s5 = _swinnerton_dyer(5)
    shifted = dup_shift(s5, 1, ZZ)
    scaled = dup_primitive([c*2**(dup_degree(shifted) - i) for i, c in enumerate(shifted)], ZZ)[1]
    f = dup_mul(s5, scaled, ZZ)
    assert dup_zz_van_hoeij(f) == sorted([s5, scaled], key=lambda g: (len(g), g))


def test_van_hoeij_against_sympy() -> None:
    rng = random.Random(21)
    checked = 0
    for trial in range(120):
        f: list[int] = [1]
        kind = trial % 3
        for _ in range(rng.randint(8, 16) if kind == 0 else rng.randint(1, 4)):
            if kind == 0:
                # many linear factors: the lattice ends as the identity
                g = [rng.randint(1, 3), rng.randint(-40, 40)]
            elif kind == 1:
                g = dup_shift(_swinnerton_dyer(rng.randint(2, 3)), rng.randint(-3, 3), ZZ)
                scale = rng.choice([1, 2, 3])
                g = [c*scale**i for i, c in enumerate(g)]
            else:
                g = [rng.randint(1, 5)] + [rng.randint(-9, 9) for _ in range(rng.randint(1, 7))]
            f = dup_mul(f, g, ZZ)
        f = dup_primitive(f, ZZ)[1]
        if dup_degree(f) < 2 or not dup_sqf_p(f, ZZ):
            continue
        checked += 1
        factors = dup_zz_van_hoeij(f)
        assert factors == _sympy_factors(f)
        product = [1]
        for g in factors:
            product = dup_mul(product, g, ZZ)
        assert product == f
    assert checked > 80
