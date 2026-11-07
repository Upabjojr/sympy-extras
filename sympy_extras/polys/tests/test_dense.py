"""The typed dense polynomial layer against SymPy's polynomial routines
(the oracle) on random inputs, and on hand-made cases."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy import Matrix, Poly, symbols, discriminant, gcd, resultant, sqf_list
from sympy.polys.domains import ZZ as SZZ, QQ as SQQ

from sympy_extras.polys.densearith import dup_add, dup_div, dup_exquo, dup_mul, dup_pdiv, dup_pow, dup_sub
from sympy_extras.polys.densebasic import dup_from_list, dup_strip
from sympy_extras.polys.densetools import (dup_clear_denoms, dup_compose, dup_diff, dup_eval, dup_integrate,
                                            dup_primitive, dup_shift, dup_trunc)
from sympy_extras.polys.domains import GF, QQ, ZZ, FractionField, PolynomialRing
from sympy_extras.polys.euclidtools import (dup_discriminant, dup_gcd, dup_gcdex, dup_invert, dup_lcm,
                                             dup_qq_gcd, dup_resultant, dup_subresultants)
from sympy_extras.polys.sqfreetools import dup_sqf_list, dup_sqf_p, dup_sqf_part

x, y = symbols('x y')


def _random_zz(rng: random.Random, degree: int, bound: int = 9) -> list[int]:
    return dup_strip([rng.randint(-bound, bound) for _ in range(degree + 1)], ZZ)


def _random_qq(rng: random.Random, degree: int) -> list[Fraction]:
    return dup_strip([Fraction(rng.randint(-9, 9), rng.randint(1, 5)) for _ in range(degree + 1)], QQ)


def _sylvester_determinant(f: list[int], g: list[int]) -> int:
    """The resultant by its definition: the determinant of the Sylvester
    matrix (``deg g`` shifted rows of ``f``, then ``deg f`` rows of ``g``)."""
    m, n = len(f) - 1, len(g) - 1
    size = m + n
    rows = []
    for i in range(n):
        rows.append([0]*i + f + [0]*(size - m - 1 - i))
    for i in range(m):
        rows.append([0]*i + g + [0]*(size - n - 1 - i))
    return int(Matrix(rows).det()) if size else 1


def _sympy_zz(f: list[int]) -> Poly:
    return Poly(f, x, domain=SZZ)


def _sympy_qq(f: list[Fraction]) -> Poly:
    return Poly([SQQ(c.numerator, c.denominator) for c in f], x, domain=SQQ)


def _from_sympy_zz(p: Poly) -> list[int]:
    return dup_strip([int(c) for c in p.all_coeffs()], ZZ)


def test_arithmetic_against_sympy() -> None:
    rng = random.Random(1)
    for _ in range(100):
        f, g = _random_zz(rng, rng.randint(0, 6)), _random_zz(rng, rng.randint(0, 6))
        F, G = _sympy_zz(f), _sympy_zz(g)
        assert dup_add(f, g, ZZ) == _from_sympy_zz(F + G)
        assert dup_sub(f, g, ZZ) == _from_sympy_zz(F - G)
        assert dup_mul(f, g, ZZ) == _from_sympy_zz(F*G)
        assert dup_pow(f, 3, ZZ) == _from_sympy_zz(F**3)
        if g:
            q, r = dup_pdiv(f, g, ZZ)
            Q, R = F.pdiv(G)
            assert (q, r) == (_from_sympy_zz(Q), _from_sympy_zz(R)), (f, g)
            # the defining identity of the pseudo-division
            scale = g[0]**max(len(f) - len(g) + 1, 0) if len(f) >= len(g) else 1
            assert dup_add(dup_mul(q, g, ZZ), r, ZZ) == [scale*c for c in f]
        assert dup_diff(f, 1, ZZ) == _from_sympy_zz(F.diff(x))
        assert dup_eval(f, 3, ZZ) == int(F.eval(3))
        assert dup_shift(f, 2, ZZ) == _from_sympy_zz(F.shift(2))
        assert dup_compose(f, g, ZZ) == _from_sympy_zz(F.compose(G)) if g else True


def test_field_division_against_sympy() -> None:
    rng = random.Random(2)
    for _ in range(60):
        f, g = _random_qq(rng, rng.randint(0, 6)), _random_qq(rng, rng.randint(0, 5))
        if not g:
            continue
        q, r = dup_div(f, g, QQ)
        Q, R = _sympy_qq(f).div(_sympy_qq(g))
        assert _sympy_qq(q) == Q and _sympy_qq(r) == R
        assert dup_add(dup_mul(q, g, QQ), r, QQ) == f
    F = GF(7)
    for _ in range(60):
        fm = dup_strip([rng.randint(0, 6) for _ in range(rng.randint(1, 7))], F)
        gm = dup_strip([rng.randint(0, 6) for _ in range(rng.randint(1, 5))], F)
        if not gm:
            continue
        qm, rm = dup_div(fm, gm, F)
        Q, R = Poly(fm, x, modulus=7).div(Poly(gm, x, modulus=7))
        assert [int(c) % 7 for c in Q.all_coeffs()] == qm or (not qm and Q.is_zero)
        assert dup_add(dup_mul(qm, gm, F), rm, F) == fm


def test_gcd_resultant_against_sympy() -> None:
    rng = random.Random(3)
    for _ in range(80):
        h = _random_zz(rng, rng.randint(0, 3), 4)
        f = dup_mul(_random_zz(rng, rng.randint(0, 4)), h, ZZ)
        g = dup_mul(_random_zz(rng, rng.randint(0, 4)), h, ZZ)
        F, G = _sympy_zz(f), _sympy_zz(g)
        assert dup_gcd(f, g, ZZ) == _from_sympy_zz(gcd(F, G)) if f or g else dup_gcd(f, g, ZZ) == []
        if f and g:
            assert dup_lcm(f, g, ZZ) == _from_sympy_zz(F.lcm(G))
            # SymPy returns the resultant with the higher degree polynomial
            # first whatever the order of the arguments, which is off by
            # the sign (-1)**(deg f deg g) when deg f < deg g (sympy-extras#25)
            expected_resultant = int(resultant(F, G))
            if len(f) < len(g) and ((len(f) - 1)*(len(g) - 1)) % 2:
                expected_resultant = -expected_resultant
            assert dup_resultant(f, g, ZZ) == expected_resultant
            assert dup_resultant(f, g, ZZ) == _sylvester_determinant(f, g)
            assert dup_discriminant(f, ZZ) == int(discriminant(F)) if len(f) > 1 else True
            # the last subresultant is the gcd up to a constant
            last = dup_subresultants(f, g, ZZ)[-1]
            assert dup_primitive(last, ZZ)[1] in (dup_primitive(dup_gcd(f, g, ZZ), ZZ)[1],
                                                  [-c for c in dup_primitive(dup_gcd(f, g, ZZ), ZZ)[1]])
        fq = [Fraction(c, 3) for c in f]
        gq = [Fraction(c, 2) for c in g]
        if fq or gq:
            expected = gcd(_sympy_qq(fq), _sympy_qq(gq))
            assert _sympy_qq(dup_gcd(fq, gq, QQ)) == expected
            assert dup_qq_gcd(fq, gq) == dup_gcd(fq, gq, QQ)


def test_extended_euclid_and_inverse() -> None:
    rng = random.Random(4)
    for _ in range(40):
        f, g = _random_qq(rng, rng.randint(1, 5)), _random_qq(rng, rng.randint(1, 5))
        s, t, h = dup_gcdex(f, g, QQ)
        assert dup_add(dup_mul(s, f, QQ), dup_mul(t, g, QQ), QQ) == h
        assert h == dup_gcd(f, g, QQ)
    F = GF(11)
    for _ in range(40):
        gm = dup_strip([rng.randint(0, 10) for _ in range(4)], F)
        fm = dup_strip([rng.randint(0, 10) for _ in range(3)], F)
        if len(gm) < 2 or not fm or len(dup_gcd(fm, gm, F)) > 1:
            continue
        inverse = dup_invert(fm, gm, F)
        assert dup_div(dup_mul(fm, inverse, F), gm, F)[1] == [1]


def test_squarefree_against_sympy() -> None:
    rng = random.Random(5)
    for _ in range(40):
        f = [1]
        for _ in range(rng.randint(1, 3)):
            f = dup_mul(f, dup_pow(_random_zz(rng, rng.randint(1, 2), 3), rng.randint(1, 3), ZZ), ZZ)
        if not f:
            continue
        content, factors = dup_sqf_list(f, ZZ)
        expected_content, expected = sqf_list(_sympy_zz(f))
        assert content == int(expected_content)
        assert [(g, k) for g, k in factors] == [(_from_sympy_zz(p), k) for p, k in expected]
        assert dup_sqf_part(f, ZZ) == _from_sympy_zz(_sympy_zz(f).sqf_part())
        assert dup_sqf_p(f, ZZ) == all(k == 1 for _, k in expected)
    # over GF(p): the p-th powers
    F = GF(3)
    for _ in range(40):
        f = [1]
        for _ in range(rng.randint(1, 3)):
            base = dup_strip([rng.randint(0, 2) for _ in range(rng.randint(2, 3))], F)
            if len(base) < 2:
                continue
            f = dup_mul(f, dup_pow(base, rng.randint(1, 4), F), F)
        if len(f) < 2:
            continue
        content, factors = dup_sqf_list(f, F)
        product = [content]
        for g, k in factors:
            product = dup_mul(product, dup_pow(g, k, F), F)
            assert dup_sqf_p(g, F)
        assert product == f
        expected = Poly(f, x, modulus=3).sqf_list()[1]
        assert sorted(k for _, k in factors) == sorted(k for _, k in expected)


def test_multivariate_through_the_tower() -> None:
    R = PolynomialRing(ZZ, 'y')
    S = PolynomialRing(R, 'x')
    # (x + y)(x - y) and (x + y)**2 in ZZ[y][x]
    f = S.mul([[1], [1, 0]], [[1], [-1, 0]])
    g = S.mul([[1], [1, 0]], [[1], [1, 0]])
    assert S.to_sympy(f).expand() == (x**2 - y**2)
    assert dup_gcd(f, g, R) == [[1], [1, 0]]
    # resultant in x of x**2 - y and x - y**2 is y**4 - y
    res = dup_resultant([[1], [], [-1, 0]], [[1], [-1, 0, 0]], R)
    assert R.to_sympy(res) == y**4 - y
    assert R.to_sympy(res) == resultant(x**2 - y, x - y**2, x)
    rng = random.Random(6)
    for _ in range(20):
        h = [[rng.randint(-2, 2) for _ in range(2)] for _ in range(2)]
        h = dup_strip([dup_strip(c, ZZ) for c in h], R)
        a = dup_strip([dup_strip([rng.randint(-2, 2) for _ in range(2)], ZZ) for _ in range(3)], R)
        b = dup_strip([dup_strip([rng.randint(-2, 2) for _ in range(2)], ZZ) for _ in range(2)], R)
        f, g = S.mul(a, h), S.mul(b, h)
        if not f or not g:
            continue
        found = S.to_sympy(dup_gcd(f, g, R))
        expected = gcd(S.to_sympy(f), S.to_sympy(g))
        assert (found - expected).expand() == 0 or (found + expected).expand() == 0, (f, g)
        assert R.to_sympy(dup_resultant(f, g, R)) == resultant(S.to_sympy(f), S.to_sympy(g), x)


def test_fraction_field_and_tools() -> None:
    F = FractionField(PolynomialRing(ZZ, 'x'))
    a = F.new([1, 0, -1], [1, 1])
    assert a.numer == [1, -1] and a.denom == [1]
    b = F.new([1], [2, 0])
    assert F.to_sympy(F.add(a, b)) == (x - 1 + 1/(2*x)).together()
    assert F.mul(b, F.invert(b)) == F.one
    assert dup_clear_denoms([Fraction(1, 2), Fraction(1, 3)], QQ) == (6, [Fraction(3), Fraction(2)])
    assert dup_trunc([7, 8, 9], 5, ZZ) == [2, -2, -1]
    assert dup_trunc([-7, -8, -9], 5, ZZ) == [-2, 2, 1]
    assert dup_integrate([Fraction(3), Fraction(0), Fraction(-2)], 1, QQ) == [Fraction(1), Fraction(0), Fraction(-2), Fraction(0)]
    assert dup_exquo([1, 0, -1], [1, 1], ZZ) == [1, -1]
    assert dup_from_list([0, 2, 3], ZZ) == [2, 3]
