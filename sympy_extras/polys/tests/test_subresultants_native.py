"""Tests of the native principal subresultant coefficients
(:mod:`sympy_extras.polys.subresultants`): against the determinants of the
Sylvester submatrices, and in several variables against the
specializations; the SymPy face is tested in ``test_euclidtools.py``."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy_extras._testing import untyped
from sympy_extras.polys.domains import GF, QQ, ZZ
from sympy_extras.polys.domains.polynomialring import PolynomialRing
from sympy_extras.polys.rings import PolyElement, PolyRing
from sympy_extras.polys.subresultants import dup_psc, psc


def _det(rows: list[list[Fraction]]) -> Fraction:
    """The determinant by Gaussian elimination over the rationals."""
    a = [list(r) for r in rows]
    n = len(a)
    result = Fraction(1)
    for i in range(n):
        pivot = next((k for k in range(i, n) if a[k][i]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            result = -result
        result *= a[i][i]
        for k in range(i + 1, n):
            factor = a[k][i]/a[i][i]
            a[k] = [u - factor*v for u, v in zip(a[k], a[i])]
    return result


def _psc_det(f: list[int], g: list[int]) -> list[Fraction]:
    """psc_j as the determinant of the Sylvester submatrix: m - j rows of
    shifts of f, n - j of g, the first n + m - 2j columns."""
    n, m = len(f) - 1, len(g) - 1
    if n < m:
        f, g, n, m = g, f, m, n
    out = []
    for j in range(m + 1):
        width = n + m - j
        rows = [[Fraction(0)]*i + [Fraction(c) for c in f] + [Fraction(0)]*(width - n - 1 - i) for i in range(m - j)]
        rows += [[Fraction(0)]*i + [Fraction(c) for c in g] + [Fraction(0)]*(width - m - 1 - i) for i in range(n - j)]
        out.append(_det([r[:n + m - 2*j] for r in rows]) if rows else Fraction(1))
    return out


def test_dup_psc_known() -> None:
    assert dup_psc([1, 0, 1], [1, 0, -1], ZZ) == [4, 0, 1]
    assert dup_psc([1, 0, -1, 0], [1, 0, -1], ZZ) == [0, 0, 1]
    assert dup_psc([1, -3, 2, 0], [1, 0, -1, 0], ZZ) == [0, 0, 3, 1]
    assert dup_psc([1, 0, 0, 1, 1], [3, 0, -2], ZZ) == [115, -27, 9]
    assert dup_psc([], [1], ZZ) == [] and dup_psc([3], [5], ZZ) == [1]
    # x**2*y + x and x + y over ZZ[y]: the resultant y**3 - y
    assert dup_psc([[1, 0], [1], []], [[1], [1, 0]], PolynomialRing(ZZ, 'y')) == [[1, 0, -1, 0], [1]]


def test_dup_psc_against_sylvester() -> None:
    rng = random.Random(3)
    for _ in range(60):
        f = [rng.randint(1, 4)] + [rng.randint(-4, 4) for _ in range(rng.randint(0, 6))]
        g = [rng.randint(1, 4)] + [rng.randint(-4, 4) for _ in range(rng.randint(0, 6))]
        if rng.random() < 0.3:
            common = [1, rng.randint(-2, 2)]
            f = [sum(f[i]*common[k - i] for i in range(len(f)) if 0 <= k - i < 2) for k in range(len(f) + 1)]
            g = [sum(g[i]*common[k - i] for i in range(len(g)) if 0 <= k - i < 2) for k in range(len(g) + 1)]
        expected = _psc_det(f, g)
        assert dup_psc(f, g, ZZ) == expected, (f, g)
        # f/2: the m - j rows of f (m = deg g) in the j-th Sylvester submatrix
        scaled = [e*Fraction(1, 2)**(len(g) - 1 - j) for j, e in enumerate(expected)]
        assert dup_psc([Fraction(c, 2) for c in f], [Fraction(c) for c in g], QQ) == scaled, (f, g)
        p = 7
        assert dup_psc([c % p for c in f], [c % p for c in g], GF(p)) == [int(e) % p for e in expected], (f, g)


def test_psc_on_rings() -> None:
    R = PolyRing(ZZ, ['x'])
    X, = R.gens
    assert psc(R.zero, R.zero) == [] and psc(X**2 + R.one, R.zero) == []
    assert psc(R.from_int(3), R.from_int(5)) == [R.one]
    assert [str(c) for c in psc(X**5 + X**3 - R.one, (X**3).mul_ground(2))] == ['32', '0', '0', '4']
    S = PolyRing(ZZ, ['x', 'y'])
    X, Y = S.gens
    assert [str(c) for c in psc(X**2*Y + X, X + Y)] == ['y**3 - y', '1']
    assert [str(c) for c in psc(X**2 - Y, X**2*Y - S.one)] == ['y**4 - 2*y**2 + 1', '0', '1']
    f = X**2 + Y**2 - S.one
    assert [str(c) for c in psc(f, f.diff(0))] == ['4*y**2 - 4', '2']
    try:
        untyped(psc)(X, PolyRing(QQ, ['x', 'y']).gens[0])
    except ValueError:
        pass
    else:
        raise AssertionError('ValueError expected')


def test_psc_commutes_with_specialization() -> None:
    # with the leading coefficients nonzero at y = a, psc_j(f, g)(a) = psc_j(f(a), g(a))
    rng = random.Random(5)
    S = PolyRing(ZZ, ['x', 'y', 'z'])
    for _ in range(30):
        def rand(degree: int) -> PolyElement[int]:
            terms = {(degree, 0, 0): rng.randint(1, 3)}
            for _ in range(rng.randint(1, 5)):
                terms[(rng.randint(0, degree - 1), rng.randint(0, 2), rng.randint(0, 2))] = rng.randint(-3, 3)
            return S.from_dict({m: c for m, c in terms.items() if c})
        f, g = rand(rng.randint(1, 4)), rand(rng.randint(1, 3))
        coefficients = psc(f, g)
        assert all(c.degree_in(0) <= 0 for c in coefficients)
        a, b = rng.randint(-3, 3), rng.randint(-3, 3)
        fa, ga = f.evaluate(1, a).evaluate(2, b), g.evaluate(1, a).evaluate(2, b)
        assert [c.evaluate(1, a).evaluate(2, b) for c in coefficients] == psc(fa, ga), (f, g, a, b)
