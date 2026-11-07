"""Sparse multivariate polynomials against SymPy's ring elements (the
oracle) and the division algorithm's identities."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy import QQ as SQQ, symbols
from sympy.polys.orderings import grevlex as sympy_grevlex, lex as sympy_lex
from sympy.polys.rings import ring
from sympy.testing.pytest import raises

from sympy_extras.polys.domains import QQ, ZZ
from sympy_extras.polys.rings import (PolyElement, PolyRing, grevlex, grlex, lex, monomial_div, monomial_divides,
                                      monomial_gcd, monomial_lcm, monomial_mul)
from sympy_extras.polys.interop import ring_from_sympy, ring_to_sympy

x, y, z = symbols('x y z')


def _random(rng: random.Random, R: PolyRing[Fraction], terms: int, degree: int) -> PolyElement[Fraction]:
    result = R.zero
    for _ in range(terms):
        monomial = tuple(rng.randint(0, degree) for _ in range(R.ngens))
        result = result + R.term(monomial, Fraction(rng.randint(-5, 5)))
    return result


def test_monomials() -> None:
    assert monomial_mul((1, 2), (3, 0)) == (4, 2)
    assert monomial_div((4, 2), (3, 0)) == (1, 2)
    raises(ValueError, lambda: monomial_div((1, 2), (3, 0)))
    assert monomial_divides((1, 0), (1, 2)) and not monomial_divides((2, 0), (1, 2))
    assert monomial_lcm((1, 2), (3, 0)) == (3, 2) and monomial_gcd((1, 2), (3, 0)) == (1, 0)
    # the orders on x**2 y versus x y**3: lex prefers the power of x,
    # the graded orders the total degree
    assert lex((2, 1)) > lex((1, 3)) and grlex((2, 1)) < grlex((1, 3)) and grevlex((2, 1)) < grevlex((1, 3))
    # grevlex: among x**2 y z and x y**2 z (same degree), the one with the
    # smaller power of the last variable is larger; here equal, then y
    assert grevlex((2, 1, 1)) > grevlex((1, 2, 1))
    assert grevlex((1, 1, 2)) < grevlex((1, 2, 1))


def test_arithmetic_against_sympy() -> None:
    rng = random.Random(1)
    R = PolyRing(QQ, ['x', 'y', 'z'], grevlex)
    S, X, Y, Z = ring('x,y,z', SQQ, sympy_grevlex)
    for _ in range(40):
        f, g = _random(rng, R, 4, 3), _random(rng, R, 3, 2)
        F, G = S.from_expr(ring_to_sympy(f)), S.from_expr(ring_to_sympy(g))
        assert ring_to_sympy((f + g)) == (F + G).as_expr()
        assert ring_to_sympy((f - g)) == (F - G).as_expr()
        assert ring_to_sympy((f*g)) == (F*G).as_expr()
        assert ring_to_sympy((f**2)) == (F**2).as_expr()
        if f.terms:
            assert f.LM == F.LM and f.LC == Fraction(int(F.LC.numerator), int(F.LC.denominator))
            assert f.monic().LC == 1 and ring_to_sympy(f.monic()) == F.monic().as_expr()
        assert ring_to_sympy(f.diff(0)) == F.diff(X).as_expr()
        assert ring_from_sympy(R, ring_to_sympy(f)) == f
    # lex order leading terms
    R2 = PolyRing(QQ, ['x', 'y'], lex)
    S2, X2, Y2 = ring('x,y', SQQ, sympy_lex)
    f2 = ring_from_sympy(R2, x*y**3 + x**2*y + y**5)
    assert f2.LM == (2, 1) and S2.from_expr(x*y**3 + x**2*y + y**5).LM == (2, 1)
    assert [m for m in f2.monomials()] == [(2, 1), (1, 3), (0, 5)]


def test_division_algorithm() -> None:
    rng = random.Random(2)
    R = PolyRing(QQ, ['x', 'y'], grevlex)
    for _ in range(40):
        f = _random(rng, R, 5, 3)
        divisors = [_random(rng, R, 2, 2) for _ in range(rng.randint(1, 3))]
        divisors = [g for g in divisors if g.terms]
        if not divisors:
            continue
        quotients, remainder = f.div(divisors)
        total = remainder
        for q, g in zip(quotients, divisors):
            total = total + q*g
        assert total == f
        # no term of the remainder is divisible by a leading monomial
        assert all(not monomial_divides(g.LM, m) for g in divisors for m in remainder.terms)
        # an exact quotient
        product = f*divisors[0]
        assert product.exquo(divisors[0]) == f
    X, Y = R.gens
    raises(ValueError, lambda: (X*Y + R.one).exquo(X))
    # the textbook example (Cox, Little, O'Shea 2.3, Example 1 in lex):
    # x**2 y + x y**2 + y**2 divided by (x y - 1, y**2 - 1)
    L = PolyRing(QQ, ['x', 'y'], lex)
    X, Y = L.gens
    quotients, remainder = (X**2*Y + X*Y**2 + Y**2).div([X*Y - L.one, Y**2 - L.one])
    assert [str(q) for q in quotients] == ['x + y', '1'] and str(remainder) == 'x + y + 1'


def test_over_the_integers_and_printing() -> None:
    R = PolyRing(ZZ, ['x', 'y'], grevlex)
    X, Y = R.gens
    f = (X*X).mul_ground(2) - Y.mul_ground(4) + R.from_int(6)
    assert str(f) == '2*x**2 - 4*y + 6'
    assert f.primitive() == (2, X*X - Y.mul_ground(2) + R.from_int(3))
    assert str(R.zero) == '0' and str(-X) == '-x' and str(X*Y - R.one) == 'x*y - 1'
    assert f.evaluate(1, ZZ.from_int(2)) == (X*X).mul_ground(2) - R.from_int(2)
    assert f.degree_in(0) == 2 and f.degree_in(1) == 1 and R.zero.degree() == -1
    assert f.coeff((2, 0)) == 2 and f.coeff((5, 5)) == 0
    assert R.with_order(lex).order == lex and R != R.with_order(lex)
    assert f.is_ground is False and R.from_int(3).is_ground is True
    assert list(f) == [((2, 0), 2), ((0, 1), -4), ((0, 0), 6)]

