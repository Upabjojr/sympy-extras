"""Tests of the factorisation over number fields (Trager): the product of
the factors is the polynomial, and the number of irreducible factors is
SymPy's (``factor_list`` with an extension, the oracle)."""
from __future__ import annotations

from fractions import Fraction

import sympy

from sympy_extras.core import I, Integer, sqrt
from sympy_extras.polys.algebraicfactor import AlgebraicPolynomialRing, ext_factor_list, ext_gcd
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.rings import PolyElement, PolyRing

SQRT2 = AlgebraicField([Fraction(1), Fraction(0), Fraction(-2)], sqrt(Integer(2)))
GAUSS = AlgebraicField([Fraction(1), Fraction(0), Fraction(1)], I)


def _product(lc: list[Fraction], factors: list[tuple[PolyElement[list[Fraction]], int]],
             ring: PolyRing[list[Fraction]]) -> PolyElement[list[Fraction]]:
    result = ring.ground(lc)
    for f, k in factors:
        result = result * f**k
    return result


def _sympy_count(expression: sympy.Expr, extension: sympy.Expr) -> int:
    _, factors = sympy.factor_list(expression, extension=extension)
    return sum(k for _, k in factors)


def test_factors_over_the_square_root_of_two() -> None:
    R = PolyRing(SQRT2, ['x', 'y'])
    x, y = R.gens
    s = R.ground(SQRT2.alpha)
    two = R.ground(SQRT2.from_int(2))
    X, Y = sympy.symbols('x y')
    cases = [(x * x - two * y * y, X**2 - 2 * Y**2), ((x**4 + R.one) * y, (X**4 + 1) * Y),
             ((x - s * y)**2 * (x + y + R.one), (X - sympy.sqrt(2) * Y)**2 * (X + Y + 1)),
             (x**2 * y**3 + y, X**2 * Y**3 + Y)]
    for f, expression in cases:
        lc, factors = ext_factor_list(f)
        assert _product(lc, factors, R) == f
        assert all(g.is_monic for g, _ in factors)
        assert sum(k for _, k in factors) == _sympy_count(expression, sympy.sqrt(2)), expression


def test_factors_over_the_gaussian_numbers() -> None:
    R = PolyRing(GAUSS, ['x', 'y'])
    x, y = R.gens
    X, Y = sympy.symbols('x y')
    for f, expression in [(x * x + y * y, X**2 + Y**2), (x**4 - y**4, X**4 - Y**4),
                          ((x * x + R.one)**3, (X**2 + 1)**3), (x**3 + y**3 + R.one, X**3 + Y**3 + 1)]:
        lc, factors = ext_factor_list(f)
        assert _product(lc, factors, R) == f
        assert sum(k for _, k in factors) == _sympy_count(expression, sympy.I), expression
    lc, factors = ext_factor_list(R.zero)
    assert lc == [] and factors == []


def test_the_gcd_over_a_number_field() -> None:
    R = PolyRing(SQRT2, ['x', 'y'])
    x, y = R.gens
    s = R.ground(SQRT2.alpha)
    a, b = (x - s * y) * (x + y)**2, (x - s * y)**2 * (x - y)
    assert ext_gcd(a, b) == x - s * y
    assert ext_gcd(a, R.zero) == a.monic() and ext_gcd(R.zero, b) == b.monic()
    S = AlgebraicPolynomialRing(SQRT2, ['x', 'y'])
    assert S.gcd(a * a, b) == (x - s * y)**2
    assert S.canonical_unit(s * x) * (s * x) == x and not S.is_negative(-x) and S.is_nonnegative(-x)
