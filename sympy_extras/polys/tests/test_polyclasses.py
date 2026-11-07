"""The typed polynomial class against SymPy's Poly (the oracle)."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy import Poly, symbols
from sympy.testing.pytest import raises

from sympy_extras.polys.domains import GF, QQ, ZZ
from sympy_extras.core import Integer as CoreInteger, sqrt as core_sqrt
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.domains.polynomialring import PolynomialRing
from sympy_extras.polys.polyclasses import DensePoly
from sympy_extras.polys.interop import dense_from_sympy, dense_to_sympy, domain_from_sympy

x, y = symbols('x y')


def _random(rng: random.Random, degree: int) -> DensePoly[int]:
    f = DensePoly.from_list([rng.randint(-5, 5) for _ in range(degree + 1)], x, ZZ)
    return f if not f.is_zero else DensePoly.from_list([1], x, ZZ)


def test_against_sympy_poly() -> None:
    rng = random.Random(1)
    for _ in range(60):
        f, g = _random(rng, rng.randint(0, 6)), _random(rng, rng.randint(1, 5))
        F, G = Poly(dense_to_sympy(f), x), Poly(dense_to_sympy(g), x)
        assert dense_to_sympy((f + g)) == (F + G).as_expr()
        assert dense_to_sympy((f - g)) == (F - G).as_expr()
        assert dense_to_sympy((f*g)) == (F*G).as_expr()
        assert dense_to_sympy((f**2)) == (F**2).as_expr()
        assert dense_to_sympy(f.diff()) == F.diff(x).as_expr()
        assert f.eval(2) == F.eval(2)
        assert dense_to_sympy(f.gcd(g)) == F.gcd(G).as_expr()
        assert f.resultant(g) == F.resultant(G) if f.degree >= g.degree else True
        q, r = f.pdiv(g)
        Q, R = F.pdiv(G)
        assert dense_to_sympy(q) == Q.as_expr() and dense_to_sympy(r) == R.as_expr()
        c, factors = f.factor_list()
        C, expected = F.factor_list()
        assert c == int(C) and [(dense_to_sympy(h), k) for h, k in factors] == sorted(
            [(h.as_expr(), k) for h, k in expected], key=lambda item: (Poly(item[0], x).degree(), Poly(item[0], x).all_coeffs(), item[1]))
        assert f.count_real_roots() == F.count_roots()
        assert f.is_squarefree == F.is_sqf
        fq = f.convert(QQ)
        assert fq.domain == QQ and dense_to_sympy(fq) == dense_to_sympy(f)
        if g.degree >= 1:
            gq = g.convert(QQ)
            qq, rq = fq.div(gq)
            assert (qq*gq + rq) == fq
            s, t, h = fq.gcdex(gq)
            assert s*fq + t*gq == h and h.is_monic


def test_multivariate_through_the_tower() -> None:
    R = PolynomialRing(ZZ, 'y')
    f = dense_from_sympy(x**2*y - y**3, x, R)
    g = dense_from_sympy(x*y + y**2, x, R)
    assert dense_to_sympy(f.gcd(g)) == x*y + y**2
    assert dense_to_sympy(f).expand() == x**2*y - y**3
    assert dense_to_sympy((f*g)).expand() == ((x**2*y - y**3)*(x*y + y**2)).expand()
    assert f.resultant(g) == domain_from_sympy(R, Poly(x**2*y - y**3, x).resultant(Poly(x*y + y**2, x)).as_expr())
    assert f.degree == 2 and f.LC == [1, 0]


def test_other_domains() -> None:
    F = GF(5)
    f = DensePoly.from_list([1, 0, 0, 0, -1], x, F)
    assert f.rep == [1, 0, 0, 0, 4]
    assert [h.rep for h, _ in f.factor_list()[1]] == [[1, 1], [1, 2], [1, 3], [1, 4]]
    raises(NotImplementedError, lambda: f.real_root_intervals())
    K = AlgebraicField([Fraction(1), Fraction(0), Fraction(-2)], core_sqrt(CoreInteger(2)))
    g = DensePoly.from_list([1, 0, -2], x, K)
    factors = g.factor_list()[1]
    assert len(factors) == 2 and all(h.degree == 1 for h, _ in factors)
    # errors
    h = DensePoly.from_list([1, 1], y, ZZ)
    raises(ValueError, lambda: f + DensePoly.from_list([1], x, ZZ))
    raises(ValueError, lambda: h + DensePoly.from_list([1], x, ZZ))
    raises(ValueError, lambda: DensePoly.from_list([1, 1], x, ZZ).exquo(DensePoly.from_list([1, 2], x, ZZ)))
    assert DensePoly.from_list([2, 4], x, ZZ).primitive() == (2, DensePoly.from_list([1, 2], x, ZZ))
    assert DensePoly.from_list([1, 2, 1], x, ZZ).sqf_list() == (1, [(DensePoly.from_list([1, 1], x, ZZ), 2)])
    assert DensePoly.from_list([1, 2, 1], x, ZZ).is_irreducible is False
    assert DensePoly.from_list([1, 0, 1], x, ZZ).is_irreducible is True
    assert DensePoly.from_list([1, 0, -2], x, ZZ).real_root_intervals(Fraction(1, 10)) == \
        [((Fraction(-23, 16), Fraction(-11, 8)), 1), ((Fraction(11, 8), Fraction(23, 16)), 1)]
